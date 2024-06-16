import os.path
import json

class Settings:
    path: str = None
    config: dict = None

    def __init__(self, path):
        self.config = {}
        self.path = path
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.config = json.load(f)

    def get(self):
        return self.config

    def set(self, config):
        if (config is None):
            config = {}

        self.config = config
        self.save()

    def has_value(self, name):
        return name in self.config

    def get_value(self, name: str, def_value: any = None):
        return self.get_value_cfg(self.config, name, def_value)

    def get_value_cfg(self, cfg: dict, name: str, def_value: any = None):
        if name not in cfg:
            cfg[name] = def_value
        return cfg[name]

    def set_value(self, name: str, value: any):
        self.config[name] = value
        self.save()

    def save(self):
        with open(self.path, 'w', encoding="utf-8") as fo:
            fo.write(json.dumps(self.config, indent=2, ensure_ascii=False))