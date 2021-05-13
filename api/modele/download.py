class CreditRisk_Download():
    def __init__(self, extension, filename, url, destination_dir):
        self.extension = extension
        self.filename = filename
        self.url = url
        self.destination_dir = destination_dir

    def download_csv(self):
        urllib.request.urlretrieve(self.url, self.destination_dir+"datarisk{}".format(self.extension))
        file_path = os.path.join(self.destination_dir, self.filename)
        return file_path
