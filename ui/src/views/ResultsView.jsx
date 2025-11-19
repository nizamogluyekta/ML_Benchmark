import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FileText, Download } from 'lucide-react';

const ResultsView = () => {
    const [reports, setReports] = useState([]);
    const [selectedReport, setSelectedReport] = useState(null);
    const [reportContent, setReportContent] = useState(null);
    const [parsedMetrics, setParsedMetrics] = useState([]);

    useEffect(() => {
        fetchReports();
    }, []);

    useEffect(() => {
        if (selectedReport && selectedReport.type === 'csv') {
            fetchReportContent(selectedReport);
        }
    }, [selectedReport]);

    const fetchReports = async () => {
        try {
            const res = await axios.get('/api/reports');
            setReports(res.data.reports);
            // Auto-select CSV summary if available
            const csvSummary = res.data.reports.find(r => r.type === 'csv');
            if (csvSummary) setSelectedReport(csvSummary);
        } catch (err) {
            console.error('Failed to fetch reports', err);
        }
    };

    const fetchReportContent = async (report) => {
        try {
            const res = await axios.get(`/api/reports/content?path=${report.path}`);
            setReportContent(res.data.content);
            if (report.type === 'csv') {
                parseCSV(res.data.content);
            }
        } catch (err) {
            console.error('Failed to fetch report content', err);
        }
    };

    const parseCSV = (csvText) => {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const data = lines.slice(1).map(line => {
            const values = line.split(',');
            const obj = {};
            headers.forEach((header, i) => {
                obj[header] = values[i];
            });
            return obj;
        });
        setParsedMetrics(data);
    };

    // Filter metrics for charting
    const chartData = parsedMetrics.map(m => ({
        name: m.model,
        rmse: parseFloat(m.rmse),
        mae: parseFloat(m.mae),
        r2: parseFloat(m.r2)
    }));

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">Benchmark Results</h2>
                <button
                    onClick={fetchReports}
                    className="text-sm text-blue-400 hover:text-blue-300 underline"
                >
                    Refresh Reports
                </button>
            </div>

            {reports.length === 0 ? (
                <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
                    <p className="text-gray-400">No reports found. Run a benchmark first.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Report List */}
                    <div className="lg:col-span-1 bg-gray-800 rounded-xl border border-gray-700 p-4">
                        <h3 className="text-lg font-semibold mb-4 text-gray-200">Available Reports</h3>
                        <div className="space-y-2">
                            {reports.map(report => (
                                <button
                                    key={report.path}
                                    onClick={() => setSelectedReport(report)}
                                    className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-colors ${selectedReport?.path === report.path
                                            ? 'bg-blue-600/20 text-blue-400 border border-blue-600/50'
                                            : 'hover:bg-gray-700 text-gray-300'
                                        }`}
                                >
                                    <FileText size={18} />
                                    <span className="truncate text-sm">{report.name}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Visualization Area */}
                    <div className="lg:col-span-2 space-y-6">
                        {selectedReport?.type === 'csv' && parsedMetrics.length > 0 && (
                            <div className="bg-gray-800 rounded-xl border border-gray-700 p-6">
                                <h3 className="text-lg font-semibold mb-6 text-gray-200">Model Comparison (RMSE)</h3>
                                <div className="h-[300px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={chartData} layout="vertical" margin={{ left: 100 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                            <XAxis type="number" stroke="#9CA3AF" />
                                            <YAxis dataKey="name" type="category" width={100} stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                                            />
                                            <Bar dataKey="rmse" fill="#3B82F6" radius={[0, 4, 4, 0]} name="RMSE" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Raw Data Table */}
                        {selectedReport?.type === 'csv' && parsedMetrics.length > 0 && (
                            <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm text-left text-gray-300">
                                        <thead className="text-xs text-gray-400 uppercase bg-gray-700/50">
                                            <tr>
                                                {Object.keys(parsedMetrics[0] || {}).slice(0, 6).map(header => (
                                                    <th key={header} className="px-6 py-3">{header}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {parsedMetrics.map((row, i) => (
                                                <tr key={i} className="border-b border-gray-700 hover:bg-gray-700/30">
                                                    {Object.values(row).slice(0, 6).map((val, j) => (
                                                        <td key={j} className="px-6 py-4">{val}</td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultsView;
