import React, { useState } from 'react';
import { LayoutDashboard, Play, Settings, Activity } from 'lucide-react';
import ConfigView from './views/ConfigView';
import RunView from './views/RunView';
import ResultsView from './views/ResultsView';

function App() {
    const [activeTab, setActiveTab] = useState('config');

    const renderContent = () => {
        switch (activeTab) {
            case 'config':
                return <ConfigView />;
            case 'run':
                return <RunView />;
            case 'results':
                return <ResultsView />;
            default:
                return <ConfigView />;
        }
    };

    return (
        <div className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
                <div className="p-6 border-b border-gray-700">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        ML Benchmark
                    </h1>
                </div>

                <nav className="flex-1 p-4 space-y-2">
                    <button
                        onClick={() => setActiveTab('config')}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${activeTab === 'config'
                                ? 'bg-blue-600 text-white'
                                : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                            }`}
                    >
                        <Settings size={20} />
                        <span>Configuration</span>
                    </button>

                    <button
                        onClick={() => setActiveTab('run')}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${activeTab === 'run'
                                ? 'bg-blue-600 text-white'
                                : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                            }`}
                    >
                        <Play size={20} />
                        <span>Run Benchmark</span>
                    </button>

                    <button
                        onClick={() => setActiveTab('results')}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${activeTab === 'results'
                                ? 'bg-blue-600 text-white'
                                : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                            }`}
                    >
                        <Activity size={20} />
                        <span>Results</span>
                    </button>
                </nav>

                <div className="p-4 border-t border-gray-700 text-xs text-gray-500 text-center">
                    v1.0.0
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-auto">
                <div className="p-8">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}

export default App;
