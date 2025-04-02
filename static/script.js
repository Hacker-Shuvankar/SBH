// Navigation functionality
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        const sectionId = this.getAttribute('data-section');
        
        // Hide all sections
        document.querySelectorAll('section').forEach(section => {
            section.style.display = 'none';
        });
        
        // Show selected section
        document.getElementById(sectionId).style.display = 'block';
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(navLink => {
            navLink.classList.remove('active');
        });
        this.classList.add('active');
    });
});

// Analysis button on home page
document.getElementById('analysis-btn').addEventListener('click', function() {
    // Hide all sections
    document.querySelectorAll('section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show analysis section
    document.getElementById('analysis').style.display = 'block';
    
    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(navLink => {
        navLink.classList.remove('active');
    });
    document.querySelector('.nav-link[data-section="analysis"]').classList.add('active');
});

// File analysis functionality
const fileInput = document.getElementById('file-input');
const fileGrid = document.getElementById('file-grid');
const popupOverlay = document.getElementById('popupOverlay');
const popup = document.getElementById('popup');
const closePopup = document.getElementById('closePopup');
const fileContent = document.getElementById('file-content');

const analyzeBtn = document.getElementById('analyze-btn');


class HESimulator {
    constructor() {
        this.apiUrl = 'http://localhost:5000/predict';
    }


}

const heSimulator = new HESimulator();

document.addEventListener("DOMContentLoaded", function () {
    // Get elements for Uploaded Files
    const fileInput = document.getElementById("file-input");
    const nof = document.getElementById("no-f");
    let noFilesMessage = document.getElementById("no-files-message");
    const fileGrid = document.getElementById("file-grid");
    const fileContent = document.getElementById("file-content");
    const popupOverlay = document.getElementById("popupOverlay");
    const closePopup = document.getElementById("closePopup");

    // Get elements for Encrypted Files
    const efileGrid = document.getElementById("efile-grid");
    const epopupOverlay = document.getElementById("epopupOverlay");
    const epopupContent = document.getElementById("efile-content");
    const eclosePopup = document.getElementById("eclosePopup");

    // File Upload Event
    if (fileInput) {
        fileInput.addEventListener("change", async function (e) {
            const file = e.target.files[0];
            if (!file) {
                alert("No file selected!");
                return;
            }

            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData,
                });

                const data = await response.json();
                alert(data.message);
                nof.style.display="none";
            } catch (error) {
                alert("Upload failed!");
                console.error("Error:", error);
            }

            const fileName = file.name;
            const text = await file.text(); // Read file content

            // Create file item
            const fileItem = document.createElement("div");
            fileItem.classList.add("file-item");
            fileItem.textContent = fileName;
            fileItem.setAttribute("data-content", text);

            // Append to grid
            fileGrid.appendChild(fileItem);

            // Click event for popup
            fileItem.addEventListener("click", function () {
                fileContent.textContent = this.getAttribute("data-content");
                popupOverlay.style.display = "flex";
            });
        });
    }

    // Close Uploaded Files Popup
    closePopup.addEventListener("click", function () {
        popupOverlay.style.display = "none";
    });

    // Close popup when clicking outside
    popupOverlay.addEventListener("click", function (event) {
        if (event.target === popupOverlay) {
            popupOverlay.style.display = "none";
        }
    });

    // Fetch Encrypted Files
    function fetchEncryptedFiles() {
        fetch("/fetch-encrypted-files")
            .then((response) => response.json())
            .then((data) => {
                efileGrid.innerHTML = ""; // Clear previous files
                if (data.files.length === 0) {
                    efileGrid.innerHTML = "<p>No encrypted files found.</p>";
                } else {
                    data.files.forEach((filename) => {
                        const fileItem = document.createElement("div");
                        fileItem.classList.add("efile-item");
                        fileItem.textContent = filename;

                        fileItem.addEventListener("click", function () {
                            fetchFileContent(filename);
                        });

                        efileGrid.appendChild(fileItem);
                    });
                }
            })
            .catch((error) => console.error("Error fetching files:", error));
    }

    // Fetch Encrypted File Content
    function fetchFileContent(filename) {
        fetch(`/fetch-encrypted-file?filename=${filename}`)
            .then((response) => response.json())
            .then((data) => {
                epopupContent.textContent = data.content || "Error loading file content.";
                epopupOverlay.style.display = "block";
            })
            .catch((error) => console.error("Error fetching file content:", error));
    }

    // Close Encrypted Files Popup
    eclosePopup.addEventListener("click", function () {
        epopupOverlay.style.display = "none";
    });

    // Close popup when clicking outside
    epopupOverlay.addEventListener("click", function (event) {
        if (event.target === epopupOverlay) {
            epopupOverlay.style.display = "none";
        }
    });

    // Fetch encrypted files initially and every 5 seconds
    fetchEncryptedFiles();
    setInterval(fetchEncryptedFiles, 5000);
});

document.getElementById("train-btn").addEventListener("click", function () {
    let progressContainer = document.getElementById("progress-container");
    let progressFill = document.getElementById("progress-fill");
    let progressText = document.getElementById("progress-text");
    let accuracyText = document.getElementById("accuracy");
    let labelsText = document.getElementById("predicted-labels");
    let predict = document.getElementById("p-btn");

    progressContainer.style.display = "block"; // Show progress bar
    progressText.innerText = "Model training in progress...";
    accuracyText.innerText = "";  // Clear previous accuracy
    labelsText.innerText = ""; // Clear previous labels
    predict.style.display="block";


    let progress = 0;
    let interval = setInterval(() => {
        if (progress >= 90) {
            clearInterval(interval);
        } else {
            progress += 10; // Increment progress
            progressFill.style.width = progress + "%";
        }
    }, 500);

    // Call Flask API for model training
    fetch("/train-model", {
        method: "POST",
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(interval); // Stop progress bar animation
        progressFill.style.width = "100%"; // Set to full
        progressText.innerText = "Model train successful!";

        // Display accuracy and predicted labels
        accuracyText.innerText = `Accuracy: ${data.accuracy}%`;
        let encryptedLabelsBox = document.getElementById("encrypted-labels");
        encryptedLabelsBox.innerText = `Encrypted Labels: ${data.Encrypted_Label?.join(", ") || "N/A"}`;
        encryptedLabelsBox.innerText = `Encrypted Labels: ${data.Encrypted_Label.join(", ")}`;

    })
    .catch(error => {
        console.error("Error:", error);
        progressText.innerText = "Error in training!";
    });
});

document.getElementById("p-btn").addEventListener("click", function () {
    let predictedLabelsBox = document.getElementById("predicted-labels");
    let labelsChartCanvas = document.getElementById("labels-chart");

    predictedLabelsBox.style.display = "block";
    labelsChartCanvas.style.display = "block"; // Show chart

    fetch("/decode", {
        method: "POST",
    })
    .then(response => response.json())
    .then(data => {
        let labels = data.predicted_labels || [];

        // Count occurrences of each label
        let labelCounts = {};
        labels.forEach(label => {
            labelCounts[label] = (labelCounts[label] || 0) + 1;
        });

        let uniqueLabels = Object.keys(labelCounts);
        let counts = Object.values(labelCounts);

        // Generate unique colors for each label
        let backgroundColors = uniqueLabels.map((_, i) => `hsl(${i * 60}, 70%, 60%)`);

        let ctx = labelsChartCanvas.getContext("2d");

        // Destroy existing chart if present
        if (window.myPieChart) {
            window.myPieChart.destroy();
        }

        // Create new pie chart
        window.myPieChart = new Chart(ctx, {
            type: "pie",
            data: {
                labels: uniqueLabels,
                datasets: [{
                    data: counts,
                    backgroundColor: backgroundColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: false, // Prevent auto-resizing
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white' // Set legend text color to white
                        }
                    }
                }
            }
        });

        // Display labels in the info box
        predictedLabelsBox.innerText = `Predicted Labels: ${labels.join(", ")}`;
    })
    .catch(error => {
        console.error("Error:", error);
        predictedLabelsBox.innerText = "Error in prediction!";
    });
});

analyzeBtn.addEventListener('click', async function() {
    const encryptedData = sessionStorage.getItem('encryptedPatient');
    if (!encryptedData) {
        alert('Please upload a valid patient data file first');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing';
    resultSection.style.display = 'none';
    
    try {
        const result = await heSimulator.predictFromEncrypted(encryptedData);
        showResults(result.risk, result.indicators);
    } catch (error) {
        console.error('Analysis error:', error);
        alert('Analysis failed. Please try again.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    }
});




// Initialize
document.getElementById('home').style.display = 'block';
document.getElementById('analysis').style.display = 'none';
document.getElementById('train').style.display = 'none';
document.getElementById('contact').style.display = 'none';
document.querySelector('.nav-link[data-section="home"]').classList.add('active');