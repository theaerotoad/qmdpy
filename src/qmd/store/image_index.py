import os
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from qmd.db import get_db_meta
from .models import encode_vector

class ImageIndexMixin:
    """Handles logic for vector indexing of multimodal images linked to documents."""

    def _cleanup_orphaned_image_data(self):
        """Removes image vectors, metadata, and physical files that are no longer referenced by any document."""
        cursor = self.conn.cursor()

        # Cleanup Image Metadata
        cursor.execute("""
            DELETE FROM image_metadata WHERE doc_hash NOT IN (SELECT DISTINCT hash FROM documents)
        """)
        
        # Cleanup Orphaned Image Vectors
        cursor.execute("SELECT rowid FROM vectors WHERE rowid NOT IN (SELECT rowid FROM chunk_metadata) AND rowid NOT IN (SELECT rowid FROM image_metadata)")
        orphan_vec_rowids = [row[0] for row in cursor.fetchall()]
        if orphan_vec_rowids:
            placeholders = ','.join(['?'] * len(orphan_vec_rowids))
            cursor.execute(f"DELETE FROM vectors WHERE rowid IN ({placeholders})", tuple(orphan_vec_rowids))
            if getattr(self, 'usearch_index', None) is not None and not self.read_only:
                for rid in orphan_vec_rowids:
                    try:
                        self.usearch_index.remove(rid)
                    except Exception:
                        pass

        # Cleanup Orphaned Images and Disk Files
        cursor.execute("SELECT hash, storage_path FROM images WHERE hash NOT IN (SELECT DISTINCT image_hash FROM image_metadata)")
        orphaned_images = cursor.fetchall()
        if orphaned_images and getattr(self.config, "image_storage_dir", None):
            storage_dir = Path(self.config.image_storage_dir).resolve()
            for img_hash, rel_path in orphaned_images:
                full_path = (storage_dir / rel_path).resolve()
                try:
                    if full_path.exists():
                        os.remove(full_path)
                        
                    # Walk up and remove empty directories until we hit the storage root
                    parent = full_path.parent
                    while parent != storage_dir:
                        try:
                            parent.rmdir()
                            parent = parent.parent
                        except OSError:
                            # rmdir throws OSError if the directory is not empty
                            break
                except Exception:
                    pass
        cursor.execute("DELETE FROM images WHERE hash NOT IN (SELECT DISTINCT image_hash FROM image_metadata)")

    def _clear_document_image_metadata(self, cursor, doc_hash: str):
        """Clears metadata and potentially orphaned vectors for a specific document hash."""
        cursor.execute("SELECT rowid FROM image_metadata WHERE doc_hash = ?", (doc_hash,))
        old_img_rowids = [r[0] for r in cursor.fetchall()]
        cursor.execute("DELETE FROM image_metadata WHERE doc_hash = ?", (doc_hash,))
        
        for r_id in old_img_rowids:
            cursor.execute("SELECT 1 FROM image_metadata WHERE rowid = ?", (r_id,))
            if not cursor.fetchone():
                cursor.execute("DELETE FROM vectors WHERE rowid = ?", (r_id,))
                if getattr(self, 'usearch_index', None) is not None and not self.read_only:
                    try: 
                        self.usearch_index.remove(r_id)
                    except Exception: 
                        pass

    def _process_document_images(self, cursor, doc_hash: str, final_chunk_texts: list, new_rowids: list, new_vectors: list):
        """Extracts qmd-img tags, computes embeddings, and writes to database."""
        if not getattr(self.config, "embed_images", False):
            return

        image_hashes = set()
        img_to_seq = {}
        for i, chunk_text in enumerate(final_chunk_texts):
            hashes = set(re.findall(r'qmd-img://([a-f0-9]{64})', chunk_text))
            for h in hashes:
                if h not in img_to_seq:
                    img_to_seq[h] = i
            image_hashes.update(hashes)
            
        if not image_hashes:
            return
        
        image_hashes = list(image_hashes)

        storage_dir = Path(self.config.image_storage_dir) if getattr(self.config, "image_storage_dir", None) else None
        if not storage_dir:
            return

        stored_quant = get_db_meta(self.conn, "vector_quantization")
        quant_type = stored_quant or getattr(self.config, "vector_quantization", "none") or "none"

        for img_hash in image_hashes:
            rel_path_img = f"{img_hash[:2]}/{img_hash[2:4]}/{img_hash[4:6]}/{img_hash}.webp"
            full_path = storage_dir / rel_path_img
            
            if not full_path.exists():
                continue
                
            # Insert into images table if not exists
            cursor.execute("SELECT hash FROM images WHERE hash = ?", (img_hash,))
            if not cursor.fetchone():
                try:
                    from PIL import Image
                    with Image.open(full_path) as img_file:
                        img_w, img_h = img_file.size
                except Exception:
                    img_w, img_h = 0, 0
                
                now_ts = datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO images(hash, storage_path, width, height, created_at) VALUES (?, ?, ?, ?, ?)",
                    (img_hash, rel_path_img, img_w, img_h, now_ts)
                )
            
            # Check if vector already exists
            cursor.execute("SELECT rowid FROM image_metadata WHERE image_hash = ? LIMIT 1", (img_hash,))
            existing_meta = cursor.fetchone()
            
            if existing_meta:
                vec_rowid = existing_meta[0]
            else:
                import base64
                webp_bytes = full_path.read_bytes()
                b64 = base64.b64encode(webp_bytes).decode('utf-8')
                b64_uri = f"data:image/webp;base64,{b64}"
                
                try:
                    img_embs = self.llm.embed_image_batch([b64_uri], show_progress=False)
                    if img_embs:
                        emb_blob = encode_vector(img_embs[0], quant_type=quant_type)
                        cursor.execute("INSERT INTO vectors(embedding) VALUES (?)", (emb_blob,))
                        vec_rowid = cursor.lastrowid
                        
                        new_rowids.append(vec_rowid)
                        new_vectors.append(img_embs[0])
                    else:
                        continue
                except Exception as e:
                    tqdm.write(f"Warning: Failed to embed image {img_hash}: {e}")
                    continue
                    
            seq_id = img_to_seq.get(img_hash, 0)
            cursor.execute(
                "INSERT INTO image_metadata(rowid, image_hash, doc_hash, seq_id) VALUES (?, ?, ?, ?)",
                (vec_rowid, img_hash, doc_hash, seq_id)
            )