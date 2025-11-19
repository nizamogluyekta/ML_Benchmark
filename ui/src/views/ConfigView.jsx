import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Save, RefreshCw } from 'lucide-react';

const ConfigView = () => {
    const [config, setConfig] = useState({ datasets: [], models: [] });
    const [availableDatasets, setAvailableDatasets] = useState([]);
    const [availableModels, setAvailableModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState(null);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [configRes, datasetsRes, modelsRes] = await Promise.all([
                axios.get('/api/config'),
                axios.get('/api/datasets'),
                axios.get('/api/models')
            ]);

            setConfig(configRes.data);
            setAvailableDatasets(datasetsRes.data.datasets);
            setAvailableModels(modelsRes.data.models);
        } catch (error) {
            console.error('Error fetching data:', error);
            setMessage({ type: 'error', text: 'Failed to load configuration data.' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        setMessage(null);
        try {
            await axios.post('/api/config', config);
            setMessage({ type: 'success', text: 'Configuration saved successfully!' });
        } catch (error) {
            console.error('Error saving config:', error);
            setMessage({ type: 'error', text: 'Failed to save configuration.' });
        } finally {
            setSaving(false);
        }
    };

    const toggleDataset = (dataset) => {
        const current = config.datasets || [];
        const updated = current.includes(dataset)
            ? current.filter(d => d !== dataset)
            : [...current, dataset];
        setConfig({ ...config, datasets: updated });
    };

    const toggleModel = (model) => {
        const current = config.models || [];
        const updated = current.includes(model)
            ? current.filter(m => m !== model)
            : [...current, model];
        setConfig({ ...config, models: updated });
    };

    // Group models by family
    const modelFamilies = availableModels.reduce((acc, model) => {
        const family = model.split('.')[0];
        if (!acc[family]) acc[family] = [];
        acc[family].push(model);
        return acc;
    }, {});

    if (loading) return <div className="text-center p-10">Loading configuration...</div>;

    return (
        <div className="space-y-8 max-w-4xl mx-auto">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">Experiment Configuration</h2>
                <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
                >
                    {saving ? <RefreshCw className="animate-spin" size={18} /> : <Save size={18} />}
                    <span>{saving ? 'Saving...' : 'Save Changes'}</span>
                </button>
            </div>

            {message && (
                <div className={`p-4 rounded-lg ${message.type === 'error' ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'}`}>
                    {message.text}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Datasets Section */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <h3 className="text-lg font-semibold mb-4 text-blue-400">Datasets</h3>
                    <div className="space-y-3">
                        {availableDatasets.map(dataset => (
                            <label key={dataset} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-700/50 cursor-pointer transition-colors">
                                <input
                                    type="checkbox"
                                    checked={(config.datasets || []).includes(dataset)}
                                    onChange={() => toggleDataset(dataset)}
                                    className="w-5 h-5 rounded border-gray-600 text-blue-600 focus:ring-blue-500 bg-gray-700"
                                />
                                <span className="text-gray-200">{dataset}</span>
                            </label>
                        ))}
                        {availableDatasets.length === 0 && (
                            <div className="text-gray-500 italic">No datasets found. Check datasets.yaml</div>
                        )}
                    </div>
                </div>

                {/* Models Section */}
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <h3 className="text-lg font-semibold mb-4 text-purple-400">Models</h3>
                    <div className="space-y-6">
                        {Object.entries(modelFamilies).map(([family, models]) => (
                            <div key={family}>
                                <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2 border-b border-gray-700 pb-1">
                                    {family}
                                </h4>
                                <div className="space-y-2">
                                    {models.map(model => (
                                        <label key={model} className="flex items-center space-x-3 p-2 rounded hover:bg-gray-700/50 cursor-pointer transition-colors">
                                            <input
                                                type="checkbox"
                                                checked={(config.models || []).includes(model)}
                                                onChange={() => toggleModel(model)}
                                                className="w-4 h-4 rounded border-gray-600 text-purple-600 focus:ring-purple-500 bg-gray-700"
                                            />
                                            <span className="text-gray-300 text-sm">{model.split('.')[1]}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ConfigView;
