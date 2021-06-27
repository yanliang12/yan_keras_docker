'''
https://keras.io/examples/structured_data/imbalanced_classification/
'''


import numpy
import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml.feature import *


sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()


##############

creditcard = sqlContext.read.option('header', True).csv('/Downloads/creditcard.csv')
creditcard.registerTempTable('creditcard')

feature_columns = [c for c in creditcard.columns if c not in ['Time', 'Class']]

sql_convert = ['double(%s) as %s'%(c,c) for c in feature_columns]

sql_convert = ', '.join(sql_convert)

creditcard1 = sqlContext.sql(u"""
	SELECT Time, %s, int(Class) as Class
	FROM creditcard
	"""%(sql_convert))
creditcard1.registerTempTable("creditcard1")

##############


assembler = VectorAssembler(
	inputCols = feature_columns,
	outputCol = "features")

features = assembler.transform(creditcard1)

##############

scaler = StandardScaler(
	inputCol="features", 
	outputCol="scaledFeatures",
	withStd=True, 
	withMean=False)

scalerModel = scaler.fit(features)

scaled_features = scalerModel.transform(features)


##############

scaled_features.registerTempTable("creditcard1")

creditcard_training_data = sqlContext.sql(u"""
	SELECT scaledFeatures.values as feature_vector, Class as label
	FROM creditcard1
	""")

creditcard_training_data.write.mode("Overwrite").json("creditcard_training_data")

##############

creditcard_training_data = sqlContext.read.json("creditcard_training_data")

scaled_features_pdf = creditcard_training_data.toPandas()

features = numpy.array([numpy.array(r) for r in scaled_features_pdf['feature_vector']])

targets = numpy.array([numpy.array([r]) for r in scaled_features_pdf['label']])

##############

numpy.save('features.npy', features)
features = numpy.load('features.npy')


numpy.save('targets.npy', targets)
targets = numpy.load('targets.npy')


num_val_samples = int(len(features) * 0.2)
train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))

##############

counts = numpy.bincount(train_targets[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]


##############


from keras.models import *
from keras.layers import *

import tensorflow
from tensorflow import keras

model = Sequential(
    [
        keras.layers.Dense(
            256, activation="relu", input_shape=(train_features.shape[-1],)
        ),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

#######

model.compile(
	optimizer=keras.optimizers.Adam(1e-2), 
	loss="binary_crossentropy",
	metrics = [
		keras.metrics.Precision(name="precision"),
		keras.metrics.Recall(name="recall"),
	])

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1}

model.fit(
    train_features,
    train_targets,
    batch_size=1024,
    epochs=5,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets),
    class_weight=class_weight,
)


model.save('fraud_model.h5')


fraud_model = keras.models.load_model('fraud_model.h5')

predictions = fraud_model.predict(features)


data = [{'score': p[0], 'label':t[0]} for p, t in zip(predictions,targets)]

prediction_df = sqlContext.read.json(sc.parallelize(data))

prediction_df.registerTempTable("prediction_df")

sqlContext.sql(u"""
	SELECT *, 
	CASE 
		WHEN score >= 0.5 THEN 1 
		ELSE 0
	END AS predicted
	FROM prediction_df
	""").registerTempTable('prediction_df1')

sqlContext.sql(u"""
	SELECT label, predicted, count(*)
	FROM prediction_df1
	GROUP BY label, predicted
	""").show()

