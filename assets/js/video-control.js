// 获取视频元素
var video1 = document.getElementById("video1");
var video2 = document.getElementById("video2");

// 当第一个视频开始播放时，启动第二个视频
video2.addEventListener("play", function() {
  video1.play();
});

// 你也可以使用 autoplay 属性来自动播放第一个视频，这样两个视频将一起播放
// video2.play();
