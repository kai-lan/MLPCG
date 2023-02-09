import logging

class LoggingWriter():
    @staticmethod
    def log(name, value): return f"{name:<30}{value:<20}\n"
    def record(self, value_dict):
        self.info = "\n" + "Basic variables\n" + '-'*50 + '\n'
        for key in value_dict:
            self.info += self.__class__.log(key, value_dict[key])
    def write(self, file):
        logging.basicConfig(filename=file, filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info(self.info)
