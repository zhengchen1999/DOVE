methods = ['EDVR', 'MuCAN', 'BasicVSR', 'IconVSR', 'TTVSR', 'BasicVSRPP']
data = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9','data10']
var methodNames = {
    'EDVR': 'EDVR',
    'MuCAN': 'MuCAN',
    'BasicVSR': 'BasicVSR',
    'IconVSR': 'IconVSR',
    'TTVSR': 'TTVSR',
    'BasicVSRPP': 'BasicVSR++'
}
var currentData = 'data1';  // Default data
var currentMethod = 'BasicVSRPP'; // Default method

var slider = document.getElementById('dataSlider');
var sliderValueDisplay = document.getElementById('sliderValue');

slider.oninput = function() {
    currentData = 'data' + this.value;
    sliderValueDisplay.textContent = this.value;
    updateVideos();
}

function updateVideos(method = null) {
    if (method) {
        currentMethod = method;
        var fullName = methodNames[method] || method;
        document.getElementById('currentMethodDisplay').textContent = fullName;
    }

    // if method is 'EDVR' and currentData is bigger than data2, then do not show the EDVR video
    // get the number of the currentData
    var currentDataNumber = parseInt(currentData.replace('data', ''));
    console.log(currentDataNumber);

    if (currentMethod == 'EDVR') {
        if (currentDataNumber == 1 || currentDataNumber == 2 || currentDataNumber == 3 || currentDataNumber == 5 || currentDataNumber == 7 || currentDataNumber == 8 || currentDataNumber == 9 || currentDataNumber == 10){
            document.getElementById('inputVideo').src = './assets/videos/' + currentData + '-input.mp4';
            document.getElementById('methodVideo').src = './assets/videos/no-video.mp4';
            document.getElementById('oursVideo').src = './assets/videos/' + currentData + '-ours.mp4';
            document.getElementById('gtVideo').src = './assets/videos/' + currentData + '-gt.mp4';
        }
        // console.log(currentDataNumber)
        // console.log(currentMethod)
        else {
            document.getElementById('inputVideo').src = './assets/videos/' + currentData + '-input.mp4';
            document.getElementById('methodVideo').src = './assets/videos/' + currentData + '-' + currentMethod + '.mp4';
            document.getElementById('oursVideo').src = './assets/videos/' + currentData + '-ours.mp4';
            document.getElementById('gtVideo').src = './assets/videos/' + currentData + '-gt.mp4';
        }
    } else {
        
        document.getElementById('inputVideo').src = './assets/videos/' + currentData + '-input.mp4';
        document.getElementById('methodVideo').src = './assets/videos/' + currentData + '-' + currentMethod + '.mp4';
        document.getElementById('oursVideo').src = './assets/videos/' + currentData + '-ours.mp4';
        document.getElementById('gtVideo').src = './assets/videos/' + currentData + '-gt.mp4';

    }
}

// Initialize
updateVideos('BasicVSRPP');

methods2 = ['base', 'tempAttn', 'FeatProp', 'AntiAlias', 'HFShuttle']
data2 = ['data1-ablation', 'data2-ablation', 'data3-ablation']
var methodNames2 = {
    'base': 'Image GigaGAN',
    'tempAttn': '+Temp Attention',
    'FeatProp': '+Feature propagation',
    'AntiAlias': '+Anti-aliasing',
    'HFShuttle': '+HF shuttle'
}
// Variables for the second set of controls
var currentData2 = 'data1-ablation';
var currentMethod2 = 'base';

var slider2 = document.getElementById('dataSlider2');
var sliderValueDisplay2 = document.getElementById('sliderValue2');

slider2.oninput = function() {
    currentData2 = 'data' + this.value + '-ablation';
    sliderValueDisplay2.textContent = this.value;
    updateVideos2();
}

function updateVideos2(method = null) {
    if (method) {
        currentMethod2 = method;
        var fullName2 = methodNames2[method] || method;
        document.getElementById('currentMethodDisplay2').textContent = fullName2;
    }

    // Update the sources for the second set of video elements
    document.getElementById('inputVideo2').src = './assets/videos/' + currentData2 + '-input.mp4';
    document.getElementById('methodVideo2').src = './assets/videos/' + currentData2 + '-' + currentMethod2 + '.mp4';
    document.getElementById('gtVideo2').src = './assets/videos/' + currentData2 + '-gt.mp4';
}

// Initialize the second set of videos
updateVideos2('base');