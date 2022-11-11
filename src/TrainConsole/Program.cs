using System;
using Microsoft.ML;
using System.Linq;

namespace TrainConsole
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @"true_car_listings.csv";
        private static string MODEL_FILEPATH = @"MLModel.zip";

        static void Main(string[] args)
        {
            // new ml context
            MLContext mLContext = new MLContext();
            // status message
            Console.WriteLine("Loading data...");
            // load train data
            IDataView trainingData = mLContext.Data.LoadFromTextFile<ModelInput>(
                path: TRAIN_DATA_FILEPATH, 
                hasHeader: true, 
                separatorChar: ','
            );
            // get train, test data holding with 20% of all
            var trainTestSplit = mLContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);
            // prepare data with pipeline
            var dataProcessPipeline = 
                // encode Make and Model columns with OneHotEncoding
                mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
                // then it concatenates Year, Milage and encoded Make and Model to Features column
                .Append(mLContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
                // normalize Features values using MinMax transform that results in a range from 0 to 1 (min: 0, max: 1)
                .Append(mLContext.Transforms.NormalizeMinMax("Features", "Features"))
                // caching in preparation for running the train *NOTE: only cache when dataset can fit into memory
                .AppendCacheCheckpoint(mLContext);
            // choose an regression model algorithm, here it is LbfgsPoissonRegression, 
            // more about algorithms in ML.net: https://docs.microsoft.com/dotnet/machine-learning/how-to-choose-an-ml-net-algorithm 
            var trainer = mLContext.Regression.Trainers.LbfgsPoissonRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            // train the model
            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
     
            // Learn more about model evaluation metrics: https://docs.microsoft.com/dotnet/machine-learning/resources/metrics
            // make predictions on train and test sets
            IDataView trainSetPredictions = model.Transform(trainTestSplit.TrainSet);
            IDataView testSetPredictions = model.Transform(trainTestSplit.TestSet);
     
            // calculate evaluation metrics for train and test sets
            var trainSetMetrics = mLContext.Regression.Evaluate(trainSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
            var testSetMetrics = mLContext.Regression.Evaluate(testSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
            // closer to 1 is better, but don't forget about overfitting and generalizing well
            Console.WriteLine($"Train Set R-Squared: {trainSetMetrics.RSquared} | Test Set R-Squared {testSetMetrics.RSquared}");
     
            // calculate crossvalidation metrics: splits data into n-partitions and trains multiple models 
            var crossValidationResults = mLContext.Regression.CrossValidate(trainingData, trainingPipeline, numberOfFolds: 5);
            var avgRSquared = crossValidationResults.Select(model => model.Metrics.RSquared).Average();
            Console.WriteLine($"Cross Validated R-Squared: {avgRSquared}");
            
            // ways to improve model:
            // - use more data
            // - use different features to train the model
            // - choose a different algorithm or update the algorithm hyperparameters
            // TODO: experiments with this

            // save model
            Console.WriteLine("Saving model...");
            mLContext.Model.Save(model, trainingData.Schema, MODEL_FILEPATH);
        }
    }
}
