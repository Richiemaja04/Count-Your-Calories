// API endpoint
const API_URL = 'http://localhost:5000';

// Global variables
let selectedFile = null;
let detectionResults = null;

// DOM elements
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const resultsSection = document.getElementById('resultsSection');
const annotatedImage = document.getElementById('annotatedImage');
const totalCalories = document.getElementById('totalCalories');
const itemCount = document.getElementById('itemCount');
const itemsList = document.getElementById('itemsList');
const voiceBtn = document.getElementById('voiceBtn');
const databaseGrid = document.getElementById('databaseGrid');
const searchInput = document.getElementById('searchInput');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadCalorieDatabase();
});

function setupEventListeners() {
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#764ba2';
        uploadBox.style.background = '#f8f9ff';
    });
    
    uploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#667eea';
        uploadBox.style.background = 'white';
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#667eea';
        uploadBox.style.background = 'white';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Change image button
    changeImageBtn.addEventListener('click', () => {
        resetUpload();
    });
    
    // Voice output button
    voiceBtn.addEventListener('click', speakSummary);
    
    // Search input
    searchInput.addEventListener('input', filterDatabase);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    uploadBox.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

async function analyzeImage() {
    if (!selectedFile) return;
    
    // Show loading state
    analyzeBtn.disabled = true;
    document.getElementById('btnText').style.display = 'none';
    document.getElementById('btnLoader').style.display = 'inline-block';
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        const response = await fetch(`${API_URL}/api/detect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Detection failed');
        }
        
        detectionResults = await response.json();
        displayResults();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze image. Make sure the server is running on localhost:5000');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        document.getElementById('btnText').style.display = 'inline';
        document.getElementById('btnLoader').style.display = 'none';
    }
}

function displayResults() {
    if (!detectionResults) return;
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Display annotated image
    annotatedImage.src = detectionResults.annotated_image;
    
    // Display summary stats
    totalCalories.textContent = `${detectionResults.total_calories} kcal`;
    itemCount.textContent = detectionResults.num_items;
    
    // Display detected items
    itemsList.innerHTML = '';
    detectionResults.detected_items.forEach((item, index) => {
        const itemCard = document.createElement('div');
        itemCard.className = 'item-card';
        itemCard.style.animationDelay = `${index * 0.1}s`;
        
        itemCard.innerHTML = `
            <div class="item-info">
                <div class="item-name">${item.name}</div>
                <div class="item-confidence">Confidence: ${item.confidence}%</div>
            </div>
            <div class="item-calories">${item.calories} kcal</div>
        `;
        
        itemsList.appendChild(itemCard);
    });
}

function speakSummary() {
    if (!detectionResults || !window.speechSynthesis) {
        alert('Speech synthesis not supported in your browser');
        return;
    }
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    let text = `Your meal contains approximately ${detectionResults.total_calories} calories. `;
    text += `I detected ${detectionResults.num_items} food items: `;
    
    detectionResults.detected_items.forEach((item, index) => {
        text += `${item.name} with ${item.calories} calories`;
        if (index < detectionResults.detected_items.length - 1) {
            text += ', ';
        }
    });
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;
    
    // Visual feedback
    voiceBtn.textContent = '🔊 Speaking...';
    voiceBtn.disabled = true;
    
    utterance.onend = () => {
        voiceBtn.textContent = '🔊 Listen to Summary';
        voiceBtn.disabled = false;
    };
    
    window.speechSynthesis.speak(utterance);
}

async function loadCalorieDatabase() {
    try {
        const response = await fetch(`${API_URL}/api/calorie-database`);
        const database = await response.json();
        
        displayDatabase(database);
    } catch (error) {
        console.error('Error loading database:', error);
        databaseGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #666;">Failed to load calorie database. Make sure the server is running.</p>';
    }
}

function displayDatabase(database) {
    databaseGrid.innerHTML = '';
    
    const items = Object.entries(database).sort((a, b) => 
        a[0].localeCompare(b[0])
    );
    
    items.forEach(([name, calories]) => {
        const dbItem = document.createElement('div');
        dbItem.className = 'db-item';
        dbItem.dataset.name = name.toLowerCase();
        
        dbItem.innerHTML = `
            <div class="db-item-name">${name}</div>
            <div class="db-item-calories">${calories} kcal</div>
        `;
        
        databaseGrid.appendChild(dbItem);
    });
}

function filterDatabase() {
    const searchTerm = searchInput.value.toLowerCase();
    const items = databaseGrid.querySelectorAll('.db-item');
    
    items.forEach(item => {
        const name = item.dataset.name;
        if (name.includes(searchTerm)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

// Handle page visibility change (stop speech if page hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.speechSynthesis) {
        window.speechSynthesis.cancel();
        voiceBtn.textContent = '🔊 Listen to Summary';
        voiceBtn.disabled = false;
    }
});