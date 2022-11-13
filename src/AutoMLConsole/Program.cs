using System;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

// new ml context
var context = new MLContext();

// load data with serialized data
var data = context.Data.LoadFromTextFile<CarData>(
    "./true_car_listings.csv",
    hasHeader: true,
    separatorChar: ','
);

// drop not used columns
var dropColumnsTransform = context.Transforms.DropColumns("Vin", "State", "City");

var newData = dropColumnsTransform.Fit(data).Transform(data);

// experiment initialization
var settings = new RegressionExperimentSettings { MaxExperimentTimeInSeconds = 60 };

var experiment = context.Auto().CreateRegressionExperiment(settings);

// show progress of experiment
// TODO: fix later
// var progress = new Progress<RunDetail<RegressionMetrics>>(p =>
// {
//     if (p.ValidationMetrics != null)
//     {
//         Console.WriteLine($"Current Run - {p.TrainerName}. R^2 - {p.ValidationMetrics.RSquared}. MAE - {p.ValidationMetrics.MeanAbsoluteError}");
//     }
// });


// TODO: and fix this
var run = experiment.Execute(newData, labelColumnName: "Price");

// var run = experiment.Execute(newData, labalColumnName: "Price", progressHandler: progress);

var bestModel = run.BestRun.Model;

var predictionEngine = context.Model.CreatePredictionEngine<CarData, CarPrediction>(bestModel);

var carData = new CarData
{
    Model = "FusionS",
    Make = "Ford",
    Mileage = 61515f,
    Year = 2012
};

var prediction = predictionEngine.Predict(carData);

Console.WriteLine($"Prediction -  {prediction.PredictedPrice:C}");
