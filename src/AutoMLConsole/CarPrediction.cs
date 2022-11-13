using Microsoft.ML.Data;

public class CarPrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}