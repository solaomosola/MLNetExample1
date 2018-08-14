using Microsoft.ML.Runtime.Api;

namespace App
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
