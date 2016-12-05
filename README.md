# ml-coding-assignment

This is the productionized code for my assignment model. 

## Usage
I originally planned to implement a command line interface and an API, but I only had enough time for the API.  This API was envisioned as accessible by a mobile application, so that an individual can enter their 'diagnosis' (i.e. injury symptoms) and stats and obtain a prediction of their ER outcome. You can access the
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
All the modules with * are implemented in this project. With another day, I would've liked to implement a productionized/automated version of the training module - the current model was pre-trained and serialized in research_and_dev.ipynb.

Implemented components:
* Scoring server
  1. neiss/neiss_server.py - very simple server implementation for demonstration purposes
* Scoring Module
* Feature Extraction Module
* Serialized Model

## Model Accuracy
