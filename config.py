from .fileutils import FileUtils


class Config:
    def __init__(self, path):
        self.path = path
        self.config = dict()

    def set(self, key, value):
        self.config[key] = value

    def get(self, key):
        if key in self.config:
            return self.config[key]
        else:
            return None

    def save(self):
        FileUtils.save_json(self.config, self.path)


