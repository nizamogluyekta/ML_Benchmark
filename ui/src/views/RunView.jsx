import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Square, Terminal } from 'lucide-react';

const RunView = () => {
    const [status, setStatus] = useState({ is_running: false, logs: [] });
    const [error, setError] = useState(null);
    const logsEndRef = useRef(null);

    useEffect(() => {
        // Poll status every second
        const interval = setInterval(fetchStatus, 1000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        // Auto-scroll to bottom of logs
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [status.logs]);

    const fetchStatus = async () => {
        try {
            const res = await axios.get('/api/status');
            setStatus(res.data);
        } catch (err) {
            console.error('Failed to fetch status', err);
        }
    };

    const handleRun = async () => {
        setError(null);
        try {
            await axios.post('/api/run');
            fetchStatus(); // Update immediately
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start benchmark');
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">Benchmark Execution</h2>
                <div className="flex items-center space-x-4">
                    {status.is_running && (
                        <span className="flex items-center text-yellow-400 text-sm font-medium animate-pulse">
                            <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
                            Running...
                        </span>
                    )}
                    <button
                        onClick={handleRun}
                        disabled={status.is_running}
                        className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${status.is_running
                                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                                : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-900/20'
                            }`}
                    >
                        {status.is_running ? <Square size={20} /> : <Play size={20} />}
                        <span>{status.is_running ? 'Benchmark in Progress' : 'Start Benchmark'}</span>
                    </button>
                </div>
            </div>

            {error && (
                <div className="bg-red-900/50 border border-red-800 text-red-200 p-4 rounded-lg">
                    {error}
                </div>
            )}

            <div className="bg-gray-900 rounded-xl border border-gray-700 overflow-hidden shadow-2xl">
                <div className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center space-x-2">
                    <Terminal size={16} className="text-gray-400" />
                    <span className="text-xs font-mono text-gray-400">Live Logs</span>
                </div>
                <div className="p-4 h-[500px] overflow-y-auto font-mono text-sm space-y-1 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
                    {status.logs.length === 0 ? (
                        <div className="text-gray-600 italic">Ready to start...</div>
                    ) : (
                        status.logs.map((log, i) => (
                            <div key={i} className="text-gray-300 whitespace-pre-wrap border-l-2 border-transparent hover:border-gray-700 pl-2">
                                {log}
                            </div>
                        ))
                    )}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>
    );
};

export default RunView;
