# wine

pip install dvc and mlflow

When the data changes:
  1. change data
  2. dvc add FILEPATH.csv
  3. git add FILEPATH.csv.dvc
  4. git commit 
  5. git tag -a VERSION -m “tag message” (+ git push origin VERSION)
  6. dvc push
  7. rm -rf DATA + DATACACHE (if you want)
