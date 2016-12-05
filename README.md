# ml-coding-assignment

This is the productionized code for my assignment model. 

## Usage
I originally planned to implement a command line interface and an API, but I only had enough time for the API.  This API was envisioned as accessible by a mobile application, so that an individual can enter his or her 'diagnosis' (i.e. injury symptoms) and stats and obtain a prediction of their ER outcome. You can access the
endpoint at http://34.192.12.23/api with GET requests corresponding to columns in the NEISS sample survey data. 

* Valid columns:
  * trmt_date - MM/DD/YYYY format
  * stratum - (single-character code)
  * age - (continuous or int)
  * sex - (int)
  * race - (int)
  * diag - (int)
  * body_part (int)
  * location (int)
  * prod1 (int)

Any invalid columns or values will be imputed with NaN values. Missing columns will also be imputed with NaN values, but the model will decrease in accuracy.

Example GET request: http://34.192.12.23/api?age=20&body_part=31&diag=62&trmt_date=11/12/2016&location=1&race=1&stratum=S&prod1=1411&sex=1

The get request returns its disposition prediction code for the input parameters.

## Building

## Project Structure
This is the overall design of the project:
![Alt text](project_organization.JPG?raw=true "Planned ML backend")
All the modules with * are implemented in this project. With another day, I would've liked to implement a productionized/automated version of the training module - **the current productionized model was pre-trained and serialized in research_and_dev.ipynb**.

Implemented components:
* Scoring server
  * **neiss/neiss_server.py** - very simple Flask server implementation for demonstration purposes
* Scoring Module (neiss/scorer/)
  * **scorer.py** - object-oriented Scorer class, used by API and CLI to make predictions from inputs
  * **scorer_constants.py** - stores constants objects, so that any changes in input module design (i.e. model codes) can propogate instantly
  * **scorer_cli.py** - UNFINSHED IMPLEMENTATION of CLI interface for scorer.py
  * **tests/*** - scoring module pytests
* Feature Extraction Module (neiss/etl_utils/)
  * **feature_extractor.py** - object-oriented FeatureExtraction class that orchestrates feature extraction. Only needs to be minorly extended for use with future implementation of training module.
  * **etl_constants.py** - stores constants objects, so that any changes in input data design (i.e. column names) can propogate instantly
  * **tests/*** feature extraction module pytests
  * **resources/***  resources necessary for feature extraction. Specifically, serialized (pickled) dictionaries of frequency counts of prod1 values during training to extract bayesian likelihood features from prod1 values in input data.
  
* Serialized Model (models/)
  * **20161205-01-43-30_rf_weighted_resampling.pkl.gz** - serialized and compressed skicit-learn model object. Currently holds a weighted resampled Random Forest trained on 12/05/2015, but it's designed to hold multiple versions and different kinds of models. Only most recent model of type [model_type] is used.

## Model Accuracy
