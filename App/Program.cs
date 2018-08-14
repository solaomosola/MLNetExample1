using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;


namespace App
{
    class Program
    {
        static void Main(string[] args)
        {

            var pipeLine = new LearningPipeline();
            string dataPath = "iris.data.txt";
            /*
             * Load the training data into the pipeline
             */
            pipeLine.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            /* Data transformation*/
            /* Microsoft says onl numbers can be processed during model training? still working*/
            pipeLine.Add(new Dictionarizer("Label"));

            /*Put all features into a vector**/
            pipeLine.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            /*Adding a learner*/
            pipeLine.Add(new StochasticDualCoordinateAscentClassifier());

            /*Convert the label back into original text after converting to number earlier*/
            pipeLine.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            /*Train the model based on the data set*/
            var model = pipeLine.Train<IrisData, IrisPrediction>();

            /*Use the trained model to make a prediction*/
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f
            });
            Console.WriteLine($"Predicted Flower is {prediction.PredictedLabels}");
        }
    }
}
