# CESPPL Dashboard Acceptance Criteria

1. A user can upload an image through the dashboard.

2. The application classifies the uploaded image and automatically saves it under the predicted class.

3. The stored filename contains the predicted class name followed by the upload date and time.

4. Every uploaded image is stored inside the SQLite database as binary data and can be fetched back without changing its original bytes.

5. The dashboard displays a tracker showing the number of stored images for each of the ten CESPPL classes, including classes that currently contain zero images.

6. A user can open any class, view all images stored under that class, download one selected image, or download the complete class collection.