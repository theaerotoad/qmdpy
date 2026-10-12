import pytest
import os
from pathlib import Path

from qmd.config import Config
from qmd.store import Store
from qmd.db import init_schema

def test_image_orphan_directory_pruning(db_conn, tmp_path):
    """
    Verify that when an image is orphaned, its physical .webp file 
    and any resulting empty parent directories are correctly pruned,
    while the root storage directory remains intact.
    """
    # Setup isolated image storage directory
    img_dir = tmp_path / ".qmd_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config(
        db_path=":memory:", 
        embed_images=True, 
        image_storage_dir=str(img_dir)
    )
    store = Store(config, connection=db_conn)
    
    # Initialize schema to ensure images and image_metadata tables exist
    init_schema(db_conn)
    
    # Generate a fake 64-char SHA-256 hash
    img_hash = "0ab873" + "0" * 58
    rel_path = f"{img_hash[:2]}/{img_hash[2:4]}/{img_hash[4:6]}/{img_hash}.webp"
    full_path = img_dir / rel_path
    
    # Create the deeply nested file structure
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(b"fake_webp_bytes")
    
    # Inject fake document and image links directly into the database
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO content (hash, body, created_at) VALUES ('doc_hash_1', 'text', 'now')")
    cursor.execute("INSERT INTO documents (collection, path, title, hash, modified_at) VALUES ('c', 'doc.md', 'Doc', 'doc_hash_1', 'now')")
    
    # Insert image tracking record
    cursor.execute(
        "INSERT INTO images (hash, storage_path, width, height, created_at) VALUES (?, ?, 100, 100, 'now')", 
        (img_hash, rel_path)
    )
    
    # Insert dummy vector blob to act as the embedding rowid
    cursor.execute("INSERT INTO vectors (embedding) VALUES (?)", (b'\x00\x00\x00\x00',))
    vec_id = cursor.lastrowid
    
    # Link the vector/image to the document
    cursor.execute(
        "INSERT INTO image_metadata (rowid, image_hash, doc_hash) VALUES (?, ?, 'doc_hash_1')", 
        (vec_id, img_hash)
    )
    db_conn.commit()
    
    # Pre-flight assertions
    assert full_path.exists(), "Test setup failed: Image file not created"
    assert full_path.parent.exists(), "Test setup failed: Image parent directory not created"
    
    # ---------------------------------------------------------
    # TRIGGER PRUNING: Delete the document referencing the image
    # ---------------------------------------------------------
    cursor.execute("DELETE FROM documents WHERE hash = 'doc_hash_1'")
    db_conn.commit()
    
    # Run the orphan cleanup routine
    store._cleanup_orphaned_data()
    
    # Verify file is deleted
    assert not full_path.exists(), "The .webp file should have been deleted."
    
    # Verify the 3-levels of empty parent directories are pruned
    assert not full_path.parent.exists(), f"Parent dir '{img_hash[4:6]}' should be deleted."
    assert not full_path.parent.parent.exists(), f"Parent dir '{img_hash[2:4]}' should be deleted."
    assert not full_path.parent.parent.parent.exists(), f"Parent dir '{img_hash[:2]}' should be deleted."
    
    # Verify the root storage directory remains perfectly intact
    assert img_dir.exists(), "The root image storage directory should NEVER be deleted."