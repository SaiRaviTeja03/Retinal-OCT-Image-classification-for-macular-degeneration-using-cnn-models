// Global variables
let alertInterval = null;
let statistics = {
    total: 0,
    normal: 0,
    abnormal: 0,
    confidences: []
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeEventListeners();
    startAlertPolling();
});

// Initialize event listeners
function initializeEventListeners() {
    const fileInput = document.getElementById('file-input');
    const browseButton = document.getElementById('browse-button');
    const dropZone = document.getElementById('drop-zone');

    if (browseButton && fileInput) {
        browseButton.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function (e) {
            const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
            if (file) {
                handleFile(file);
            }
        });
    }

    if (dropZone) {
        dropZone.addEventListener('click', function (e) {
            if (e.target.id !== 'browse-button' && fileInput) {
                fileInput.click();
            }
        });

        dropZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            dropZone.classList.add('drag-over', 'border-primary');
        });

        dropZone.addEventListener('dragleave', function (e) {
            e.preventDefault();
            dropZone.classList.remove('drag-over', 'border-primary');
        });

        dropZone.addEventListener('drop', function (e) {
            e.preventDefault();
            dropZone.classList.remove('drag-over', 'border-primary');

            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleFile(files[0]);
            }
        });
    }
}

// Handle file processing
async function handleFile(file) {
    const isValid = await validateFile(file);
    if (!isValid) return;

    showImagePreview(file);
    uploadAndClassify(file);
}

// Validate file
async function validateFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif'];
    const validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
    const maxSize = 16 * 1024 * 1024;

    const fileName = file.name.toLowerCase();
    const hasValidMime = validTypes.includes(file.type);
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));

    if (!hasValidMime && !hasValidExtension) {
        showAlert('Invalid file type. Please upload PNG, JPG, JPEG, or TIFF OCT image.', 'danger');
        return false;
    }

    if (file.size > maxSize) {
        showAlert('File too large. Maximum size is 16MB.', 'danger');
        return false;
    }

    return await validateOCTImage(file);
}

// Validate OCT image
function validateOCTImage(file) {
    return new Promise((resolve) => {
        const validationBox = document.getElementById('validation-progress');
        if (validationBox) validationBox.style.display = 'block';

        const reader = new FileReader();

        reader.onload = function (e) {
            const img = new Image();

            img.onload = function () {
                const width = img.width;
                const height = img.height;
                const aspectRatio = width / height;

                const minSize = Math.min(width, height) >= 100;
                const withinMaxSize = Math.max(width, height) <= 5000;

                if (aspectRatio < 0.2 || aspectRatio > 5.0) {
                    if (validationBox) validationBox.style.display = 'none';
                    showAlert('Invalid image aspect ratio. Please upload a standard OCT image.', 'danger');
                    resolve(false);
                    return;
                }

                if (!minSize || !withinMaxSize) {
                    if (validationBox) validationBox.style.display = 'none';
                    showAlert('Invalid image size. Images should be between 100 and 5000 pixels.', 'danger');
                    resolve(false);
                    return;
                }

                if (validationBox) validationBox.style.display = 'none';
                resolve(true);
            };

            img.onerror = function () {
                if (validationBox) validationBox.style.display = 'none';
                showAlert('Invalid image file. Please upload a valid image.', 'danger');
                resolve(false);
            };

            img.src = e.target.result;
        };

        reader.onerror = function () {
            if (validationBox) validationBox.style.display = 'none';
            showAlert('Error reading file. Please try again.', 'danger');
            resolve(false);
        };

        reader.readAsDataURL(file);
    });
}

// Show image preview
function showImagePreview(file) {
    const reader = new FileReader();
    const previewContainer = document.getElementById('image-preview');

    reader.onload = function (e) {
        previewContainer.innerHTML = `
            <img src="${e.target.result}" class="img-fluid rounded shadow" alt="OCT Image" style="max-height: 350px; object-fit: contain;">
            <div class="mt-3">
                <small class="text-muted">
                    <strong>File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${formatFileSize(file.size)}<br>
                    <strong>Type:</strong> ${file.type || 'Unknown'}
                </small>
            </div>
        `;
    };

    reader.readAsDataURL(file);
}

// Upload and classify
function uploadAndClassify(file) {
    const formData = new FormData();
    formData.append('file', file);

    const uploadProgress = document.getElementById('upload-progress');
    if (uploadProgress) uploadProgress.style.display = 'block';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (uploadProgress) uploadProgress.style.display = 'none';

            if (data.error) {
                showAlert(data.error, 'danger');
                return;
            }

            showResults(data);
            updateStatistics(data);
            showAlert(`Classification complete: ${data.class} (${(data.confidence * 100).toFixed(1)}% confidence)`, 'success');
        })
        .catch(error => {
            if (uploadProgress) uploadProgress.style.display = 'none';
            showAlert('Error uploading file: ' + error.message, 'danger');
        });
}

// Show only prediction class and confidence
function showResults(data) {
    const resultsCard = document.getElementById('results-card');
    const predictionResult = document.getElementById('prediction-result');

    let resultClass = 'success';
    let icon = 'fa-check-circle';

    if (data.class !== 'NORMAL') {
        resultClass = 'warning';
        icon = 'fa-exclamation-triangle';
    }

    predictionResult.innerHTML = `
        <div class="alert alert-${resultClass}">
            <h4 class="alert-heading mb-3">
                <i class="fas ${icon} me-2"></i>
                Prediction Class: ${data.class}
            </h4>
            <p class="mb-0">
                <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
            </p>
        </div>
    `;

    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Update statistics
function updateStatistics(data) {
    statistics.total++;
    statistics.confidences.push(data.confidence);

    if (data.class === 'NORMAL') {
        statistics.normal++;
    } else {
        statistics.abnormal++;
    }

    document.getElementById('total-images').textContent = statistics.total;
    document.getElementById('normal-count').textContent = statistics.normal;
    document.getElementById('abnormal-count').textContent = statistics.abnormal;

    const avgConf = statistics.confidences.reduce((a, b) => a + b, 0) / statistics.confidences.length;
    document.getElementById('avg-confidence').textContent = (avgConf * 100).toFixed(1) + '%';
}

// Start alert polling
function startAlertPolling() {
    loadAlerts();
    alertInterval = setInterval(loadAlerts, 5000);
}

// Load alerts
function loadAlerts() {
    fetch('/alerts')
        .then(response => response.json())
        .then(alerts => {
            displayAlerts(alerts);
        })
        .catch(error => {
            console.error('Error loading alerts:', error);
        });
}

// Display alerts without severity badge
function displayAlerts(alerts) {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="text-muted text-center">No alerts yet</div>';
        return;
    }

    container.innerHTML = alerts.map(alert => `
        <div class="alert alert-${getAlertType(alert.type)} alert-dismissible fade show mb-2" role="alert">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <i class="fas ${getAlertIcon(alert.type)} me-2"></i>
                    <strong>${alert.message}</strong>
                    <div class="small text-muted">${alert.timestamp}</div>
                </div>
            </div>
        </div>
    `).reverse().join('');
}

// Helpers
function getAlertType(type) {
    const types = {
        success: 'success',
        warning: 'warning',
        error: 'danger',
        info: 'info'
    };
    return types[type] || 'info';
}

function getAlertIcon(type) {
    const icons = {
        success: 'fa-check-circle',
        warning: 'fa-exclamation-triangle',
        error: 'fa-times-circle',
        info: 'fa-info-circle'
    };
    return icons[type] || 'fa-info-circle';
}

function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertContainer.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px; max-width: 420px;';
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    document.body.appendChild(alertContainer);

    setTimeout(() => {
        if (alertContainer.parentNode) {
            alertContainer.remove();
        }
    }, 5000);
}

function clearAlerts() {
    fetch('/clear_alerts', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(() => {
            showAlert('All alerts cleared', 'success');
            loadAlerts();
        })
        .catch(error => {
            showAlert('Error clearing alerts: ' + error.message, 'danger');
        });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Cleanup
window.addEventListener('beforeunload', function () {
    if (alertInterval) {
        clearInterval(alertInterval);
    }
});