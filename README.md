# Spark Machine Learning Library (MLlib)

## Nama : Reynaldi Fakhri Pratama
## NIM  : 2041720209
## Kelas: TI-3A

## 1. load pyspark library
```python
 # Connect Google Drive Untuk Ambil Data
 from google.colab import drive
 drive.mount('/content/drive')

 # Install PySpark
 !pip install pyspark
```
### penjelasan
- Pada tahap ini kita mengimport library pyspark untuk mengolah data yang akan kita gunakan
- Kemudian kita menghubungkan google drive dengan google colab untuk mengambil data yang akan kita gunakan

## 2. Movie Lens Recommendation

### 2.1. Load Data
```python
 # Import Library
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Movie Lens").getOrCreate()

 # Parse String Menjadi Objek Rating
def parseRating(str):
    fields = str.split(",")
    assert len(fields) == 4
    return (int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3]))


 # Baca File
raw = spark.read.text("/content/drive/MyDrive/S6/big data/ml-latest-small/ratings.dat").rdd.map(lambda x: x[0])
header = raw.first()
data = raw.filter(lambda x: x != header)
ratings = data.map(parseRating).toDF(["userId", "movieId", "rating", "timestamp"])


 # Data Training 80% dan Test 20%
training, test = ratings.randomSplit([0.8, 0.2])

 # Membuat Model
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)
model.save("mymodel")

 # Prediksi Data
predictions = model.transform(test)
mse = predictions.withColumn("diff", col("rating") - col("prediction")).select((col("diff") ** 2).alias("squared_diff")).filter(~col("squared_diff").isNull()).agg({"squared_diff": "sum"}).collect()[0][0]
print("Mean Squared Error:", mse)

predictions.show(10)

 # Menyimpan Hasil Prediksi
predictions.write.format("csv").save("ml-predictions.csv")
```
### penjelasan
kode diatas merupakan kode untuk memuat data yang akan kita gunakan, data yang kita gunakan adalah data movie lens yang berisi data rating dari film-film yang ada di movie lens. kemudian kita membagi data tersebut menjadi 2 bagian yaitu data training dan data testing. data training sebesar 80% dan data testing sebesar 20%. kemudian kita membuat model dengan menggunakan algoritma ALS (Alternating Least Squares) yang merupakan algoritma yang digunakan untuk collaborative filtering. kemudian kita melakukan prediksi data dengan menggunakan model yang telah kita buat. kemudian kita menyimpan hasil prediksi tersebut kedalam file csv.

### 2.2. balistic statistics
```python
 # Import Library
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Movie Lens").getOrCreate()
sc = spark.sparkContext

# Parse String Menjadi Objek Rating
def parseRating(str):
    fields = str.split(",")
    assert len(fields) == 4
    return (int(fields[0]), int(fields[1]), float(fields[2]), int(fields[3]))

# Baca File
raw = spark.read.text("/content/drive/MyDrive/S6/big data/ml-latest-small/ratings.csv").rdd.map(lambda x: x[0])
header = raw.first()
data = raw.filter(lambda x: x != header)
ratings = data.map(parseRating).toDF(["userId", "movieId", "rating", "timestamp"])
class Rating:
    def __init__(self, userId, movieId, rating):
        self.userId = userId
        self.movieId = movieId
        self.rating = rating

ratings_df = ratings.select(["userId", "movieId", "rating"])

# Build the recommendation model using ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(ratings_df)

# Generate product recommendations for user ID 1
products = model.recommendForUserSubset(spark.createDataFrame([(1,)]).toDF("userId"), 10)

products.show()
```
### penjelasan
kode diatas merupakan kode untuk memuat model yang telah kita buat sebelumnya. kemudian kita melakukan prediksi data dengan menggunakan model yang telah kita buat. kemudian kita menyimpan hasil prediksi tersebut kedalam file csv.

### 3.1. Load Data
```python
# Import Library
from pyspark.mllib.stat import Statistics
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Movie Lens").getOrCreate()
sc = spark.sparkContext

# Baca File
raw = spark.read.csv("/content/drive/MyDrive/S6/big data/ml-latest-small/ratings.csv", header=True)
ratings = raw.selectExpr("cast(userId as int) userId", "cast(movieId as int) movieId", "cast(rating as float) rating", "cast(timestamp as int) timestamp")

# Diambil dari ratings.csv
mat = ratings.select("rating").rdd.map(lambda x: [x[0]])

# Perhitungan Statistics
summary = Statistics.colStats(mat)
print("Mean:", summary.mean()[0])
print("Variance:", summary.variance()[0])
print("Number of Non-Zeros:", summary.numNonzeros()[0])
```

### penjelasan

kode diatas merupakan kode untuk memuat data yang akan kita gunakan, data yang kita gunakan adalah data movie lens yang berisi data rating dari film-film yang ada di movie lens. kemudian kita menghitung statistik dari data tersebut seperti mean, variance, dan number of non-zeros.

