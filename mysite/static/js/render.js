


// var request = new XMLHttpRequest();request.open('GET', "stop.json");
// request.responseType = 'json';
// request.send();


// request.onload = () => {
//     var res = request.response.dataPoints;
//     for (const key in res) {
//         frames.push(res[key]['frame']);
//         loc.push([res[key]['x'], res[key]['y']]);
//         confidence.push(res[key]['confidence']);
//     }
//     console.log(loc);
//     console.log(confidence);
//     console.log(frames);
//     frameMaker(currFrame);
// };

// const frameMaker = (num) => {
//     var video = document.getElementById('video');

//     if (confidence[num] > .2 && confidence[num] < .8) {
//         video.currentTime = (frames[num] / 24);
//         var x_pos = ((loc[num][0] / 1920) * 800);
//         var y_pos = 64 + ((loc[num][1] / 1080) * 480);
//         $('#draw-on').show();
//         $('#draw-on').css({ left: x_pos })
//         $('#draw-on').css({ top: y_pos })
//     } else {
//         currFrame += 1;
//         frameMaker(currFrame);
//     }
// };

// const calculateTimeAt = (target) => {

//     var video = document.getElementById('video');

//     switch (target) {
//         case 'picture':
//             var p_time = 0;

//             for(var i = 0; i < frames.length; i++) {
//                 if(loc[i][0] > 450 && loc[i][0] < 1350 && loc[i][1] > 60 && loc[i][1] < 850) {
//                     if((i + 1) >= frames.length) {
//                         p_time += (video.duration / 24) - frames[i];
//                     } else {
//                         p_time += frames[i+1] - frames[i];
//                     }
//                 }
//             }

//             return (p_time * 24);
//         case 'words':
//             var w_time = 0;

//             for(var i = 0; i < frames.length; i++) {
//                 if(loc[i][0] > 550 && loc[i][0] < 1450 && loc[i][1] > 850 && loc[i][1] < 1080) {
//                     if((i + 1) >= frames.length) {
//                         w_time += (video.duration / 24) - frames[i];
//                     } else {
//                         w_time += frames[i+1] - frames[i];
//                     }
//                 }
//             }

//             return (w_time * 24);
//         default:
//             break;
//     }
// };

// const checkLimit = () => {
//     if (currFrame >= frames.length) {
//         var p_time = calculateTimeAt('picture');
//         var w_time = calculateTimeAt('words');
//         $('.img-picker').replaceWith(`
        
//         <div class="final-data">

//             <h2> Time looking at picture: ${p_time} </h2>
//             <br/><br/>
//             <h2> Time looking at words: ${w_time} </h2>
//             <br/><br/>
//             <h2> Time looking at neither: ${document.getElementById('video').duration - p_time - w_time} </h2>
//             <br/><br/>

//         </div>
        
//         `);
//     }
// };

// const changePage = (event) => {
//     event.preventDefault();

//     let vid_path = $('#tracker-vid')[0].value


//     checkLimit();

//     frameMaker(currFrame);
// }
// const wrongClick = (event) => {
//     confidence[currFrame] = 0;
//     $(document).load("load_vid.html")
// }


// const rightClick = (event) => {
//     confidence[currFrame] = 1;
//     currFrame += 1;
//     currFrame += 1;

//     checkLimit();

//     frameMaker(currFrame);
// }


// $(document).on('click', '.right', rightClick);
// $(document).on('click', '.wrong', wrongClick);
// $(document).on('click', '#send-vid-path', changePage);