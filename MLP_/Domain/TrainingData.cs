using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace MLP_.Application.Training
{
    public class TrainingData
    {
        private string _filename;
        private int[] _topology;
        private List<double[]> _trainData;
        private List<double[]> _targetData;

        public TrainingData(string filename)
        {
            _filename = filename;
            _trainData = new List<double[]>();
            _targetData = new List<double[]>();

            ReadTrainData();
        }

        private void ReadTrainData()
        {
            int counter = 0;
            string line;

            string directory = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            StreamReader file = new StreamReader(directory + "../../../Application/Training/" + _filename);

            while ((line = file.ReadLine()) != null)
            {
                counter++;

                if (counter == 1)
                {
                    _topology = line.Split(',').Select(x => int.Parse(x)).ToArray();
                }
                else if (counter % 2 == 0)
                {
                    _trainData.Add(line.Split(',').Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray());
                }
                else
                {
                    _targetData.Add(line.Split(',').Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray());
                }               
            }
            
            file.Close();

        }

        public int[] getTopology()
        {
            return _topology;
        }

        public List<double[]> getTrainData()
        {
            return _trainData;
        }

        public List<double[]> getTargetData()
        {
            return _targetData;
        }
    }
}
