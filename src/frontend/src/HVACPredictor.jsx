
import { useState, useEffect } from 'react';
import axios from 'axios';
import { parseMetricsAndConfusionMatrix } from './utils/metricsParser';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  LabelList
} from 'recharts';

function App() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState(null);
  const [intervalId, setIntervalId] = useState(null);
  const [selectedTest, setSelectedTest] = useState("both");

  const runModel = async () => {
    setLoading(true);
    setProgress(0);
    setResult(null);
    setTestResult(null);
    setError(null);

    const id = setInterval(() => {
      setProgress(prev => {
        if (prev < 85) return prev + Math.random() * 10;
        if (prev < 97) return prev + Math.random();
        return prev;
      });
    }, 250);
    setIntervalId(id);

    try {
      const response = await axios.post('http://127.0.0.1:5000/run-model');
      setProgress(100);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong');
    } finally {
      clearInterval(id);
      setTimeout(() => {
        setLoading(false);
      }, 500);
    }
  };

  const runTest = async () => {
    setLoading(true);
    setProgress(0);
    setResult(null);
    setTestResult(null);
    setError(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/run-test', {
        test_type: selectedTest
      });
      setProgress(100);
      const { chartData, confusionChart } = parseMetricsAndConfusionMatrix(response.data.results);
      setTestResult({ ...response.data, chartData, confusionChart });
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong during testing');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [intervalId]);

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center px-6 py-10 bg-cyan-100 font-sans">
    <div className="bg-emerald-50 shadow-xl rounded-2xl p-8 w-full max-w-2xl text-center backdrop-blur-sm">
    <h1 className="text-3xl font-bold text-gray-800 mb-8 px-4 text-center w-full">
  HVAC Fault Detection Dashboard
</h1>
    


        <div className="flex flex-wrap justify-center gap-4 mb-10">
          <button
            onClick={runModel}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-3 rounded-full shadow-md transition-all hover:scale-105 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Training Model...' : 'Train Model'}
          </button>

          <button
            onClick={runTest}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold px-6 py-3 rounded-full shadow-md transition-all hover:scale-105 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Testing...' : 'Run Test'}
          </button>

          <select
            className="border border-gray-300 rounded-full px-5 py-2 text-gray-700 text-sm focus:outline-none focus:ring focus:border-blue-300"
            value={selectedTest}
            onChange={(e) => setSelectedTest(e.target.value)}
          >
            <option value="both">Both Datasets</option>
            <option value="fault">Fault Only</option>
            <option value="normal">Normal Only</option>
            <option value="mixed">Mixed Dataset</option>
            <option value="mixedNoGaps">MixedCorrect Dataset</option>
          </select>
        </div>

        {loading && (
          <div className="mb-8">
            <div className="w-full bg-gray-300 rounded-full h-3">
              <div
                className="bg-blue-500 h-full rounded-full transition-all duration-300 ease-in-out"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-center text-gray-600 mt-1">{Math.floor(progress)}%</p>
          </div>
        )}

        {result && (
          <div className="mt-8 bg-green-50 border border-green-200 rounded-lg p-4">
            <h2 className="text-xl font-semibold text-green-700 mb-2">Model Results</h2>
            <pre className="text-sm text-gray-800 whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}

        {testResult && (
          <div className="mt-8 flex flex-col lg:flex-row gap-8">
            {/* Left: Raw Results */}
            <div className="w-full lg:w-1/2">
              <h2 className="text-xl font-semibold text-indigo-700 mb-3">Test Results</h2>
              <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm text-gray-700 overflow-auto max-h-[500px] whitespace-pre-wrap">
                <pre>
                  {"─".repeat(60) + "\n"}
                  {testResult.results}
                  {"─".repeat(60)}
                </pre>
              </div>
            </div>

            {/* Right: Graphs */}
            <div className="w-full lg:w-1/2 space-y-6">
              <div className="bg-white p-4 border rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-2 text-blue-700">Performance Metrics (%)</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={testResult.chartData}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 50, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                    <YAxis type="category" dataKey="metric" />
                    <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                    <Bar dataKey="value" fill="#3b82f6" barSize={30}>
                      <LabelList dataKey="value" position="right" formatter={(val) => `${val.toFixed(1)}%`} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white p-4 border rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-2 text-blue-700">Confusion Matrix (Percentage)</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart
                    layout="vertical"
                    data={testResult.confusionChart.map(row => {
                      const predNorm = row["Predicted Normal"] ?? 0;
                      const predFault = row["Predicted Fault"] ?? 0;
                      const total = predNorm + predFault;
                      let correct = 0;
                      let incorrect = 0;
                      if (total > 0) {
                        if (row.actual === "Normal") {
                          correct = (predNorm / total) * 100;
                          incorrect = (predFault / total) * 100;
                        } else if (row.actual === "Fault") {
                          correct = (predFault / total) * 100;
                          incorrect = (predNorm / total) * 100;
                        }
                      }
                      return {
                        actual: row.actual,
                        Correct: parseFloat(correct.toFixed(2)),
                        Incorrect: parseFloat(incorrect.toFixed(2))
                      };
                    })}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                    <YAxis type="category" dataKey="actual" />
                    <Tooltip formatter={(val) => `${val.toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="Correct" stackId="a" fill="#22c55e" />
                    <Bar dataKey="Incorrect" stackId="a" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-8 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong className="font-bold">Error:</strong> <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

