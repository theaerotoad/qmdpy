class Colors:
    PLAIN_MODE = False
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    DIM = "\033[90m"

    @classmethod
    def set_plain_mode(cls, enabled: bool = True):
        cls.PLAIN_MODE = enabled
        if enabled:
            cls.RESET = ""
            cls.BOLD = ""
            cls.UNDERLINE = ""
            cls.CYAN = ""
            cls.GREEN = ""
            cls.YELLOW = ""
            cls.RED = ""
            cls.MAGENTA = ""
            cls.DIM = ""

c = Colors()