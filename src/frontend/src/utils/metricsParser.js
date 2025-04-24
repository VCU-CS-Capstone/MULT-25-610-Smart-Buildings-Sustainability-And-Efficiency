// utils/metricsParser.js
export function parseMetricsAndConfusionMatrix(resultsStr) {
  const lines = resultsStr.split('\n').filter(Boolean);
  const metrics = [];
  const confusionMatrix = [];

  let row1 = null;

  for (let line of lines) {
    line = line.trim();
    if (line.startsWith("Accuracy")) {
      metrics.push({ metric: "Accuracy", value: parseFloat(line.split(':')[1]) * 100 });
    } else if (line.startsWith("F1-score")) {
      metrics.push({ metric: "F1 Score", value: parseFloat(line.split(':')[1]) * 100 });
    } else if (line.startsWith("Precision")) {
      metrics.push({ metric: "Precision", value: parseFloat(line.split(':')[1]) * 100 });
    } else if (line.startsWith("Sensitivity")) {
      metrics.push({ metric: "Sensitivity", value: parseFloat(line.split(':')[1]) * 100 });
    } else if (line.startsWith("Specificity")) {
      metrics.push({ metric: "Specificity", value: parseFloat(line.split(':')[1]) * 100 });
    } else if (line.startsWith("Actual Normal")) {
      row1 = line.match(/\d+/g).map(Number);
    } else if (line.startsWith("Actual Fault") && row1) {
      const row2 = line.match(/\d+/g).map(Number);
      confusionMatrix.push({
        actual: "Normal",
        "Predicted Normal": row1[0],
        "Predicted Fault": row1[1]
      });
      confusionMatrix.push({
        actual: "Fault",
        "Predicted Normal": row2[0],
        "Predicted Fault": row2[1]
      });
    }
  }

  return { chartData: metrics, confusionChart: confusionMatrix };
}

  
  function getMetricTooltip(metric) {
    switch (metric) {
      case "Accuracy": return "(TP + TN) / Total";
      case "F1-score": return "2 * (Precision * Recall) / (Precision + Recall)";
      case "Precision": return "TP / (TP + FP)";
      case "Sensitivity (Recall)": return "TP / (TP + FN)";
      case "Specificity": return "TN / (TN + FP)";
      default: return "";
    }
  }
  