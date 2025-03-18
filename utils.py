
def extract_filename(file_path: str) -> str:
    return file_path.split('/')[-1].split('.')[0]

def extract_fileid(file_path: str) -> str:
    return file_path.split('.')[0]