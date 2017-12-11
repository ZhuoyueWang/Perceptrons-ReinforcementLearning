close all
clear all
filename = 'anime_input.csv';
M = csvread(filename);
ballx = M(:,1);
bally = M(:,2);
paddley = M(:,5);
paddleheight = 0.2;
[r,c] = size(M);
for i = 1:r
   plot(ballx(i),bally(i),'ro','MarkerSize',8)
   hold on
   plot([1,1],[paddley(i),paddley(i)+paddleheight],'b-','LineWidth',4)
   hold off
   axis([-0.01,1.01,-0.01,1.01]);
   title('Anmimation of Part 2')
   frame(i) = getframe(gcf);

    
end
movie(frame,1)
movie2avi(frame,'part2_1_animation')