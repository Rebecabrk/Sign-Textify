import kagglehub

download_location = 'data' # In current directory

# Download latest version
path = kagglehub.dataset_download("grassknoted/asl-alphabet", path=download_location)

print("Path to dataset files:", path)