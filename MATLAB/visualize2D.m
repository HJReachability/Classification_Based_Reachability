function [l]=visualize2D(g, dataCase1, dataCase2, A, B)
%% Obstacles
video = 1;

%% Neural Net Value
% V = [t, x, y]

% take min Value
%Vmin = V(end, :,:);
%Vmin = squeeze(Vmin);

%% Neural Net Grid
gNN_min = [-4; -4]; % Lower corner of computation domain
gNN_max = [4; 4];    % Upper corner of computation domain
N = [121; 121];         % Number of grid points per dimension    
gNNA = createGrid(gNN_min, gNN_max, N);

gNN_min = [-10; -10]; % Lower corner of computation domain
gNN_max = [10; 10];    % Upper corner of computation domain
N = [121; 121];         % Number of grid points per dimension    
gNNB = createGrid(gNN_min, gNN_max, N);

figure(1)
clf
%% Case 1
%subplot(1,2,1)
if video
    video_filename = ['case1_rotate_' datestr(now,'YYYYMMDD_hhmmss') '.mp4'];
    vout = VideoWriter(video_filename,'MPEG-4');
    vout.Quality = 100;
    vout.FrameRate = 80;
    vout.open;
end

% Make circle obstacles
r = .5;
x1 = (1+.5/sqrt(2));
x2 = x1;
y1 = (1+.5/sqrt(2));
y2 = -y1;
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);
hObs1 = fill(x1+xp,y1+yp,'k');
hold on
hObs2 = fill(x2+xp,y2+yp,'k');

% make square obstacle
x = [-3 -1 -1 -3];
y = [-1 -1 1 1];
hObs3 = fill(x,y,'k');

% make target set
data0x = [-1 1 1 -1];
data0y = [-1 -1 1 1];
h0 = plot(data0x,data0y);


hObs.LineWidth = 2;


h0.LineWidth = 2;
h0.LineStyle = '--';
h0.Color = 'k';
    
%axis([-3 3 -3 3])
axis([-4 4 -4 2 -1 1])
axis square
xlabel('$x$','interpreter','latex');
ylabel('$y$','interpreter','latex');
zlabel('$V(t,s)$','interpreter','latex');
%title('$u_1 \in [-1,1], u_2 \in [.5, 1]$','interpreter','latex');
set(gca,'FontSize',25)
set(gcf,'Color','w')
box on

azRange = [-180 -160];
elRange = [90 10];

length = 160;
azStep = (azRange(2)-azRange(1))/length;
elStep = (elRange(2)-elRange(1))/length;

az = azRange(1);
el = elRange(1);
axis off
while az < azRange(2) && el > elRange(2)
    az = az + azStep;
    el = el + elStep;
    view(az,el)
        if video
        current_frame = getframe(gcf); %gca does just the plot
        writeVideo(vout,current_frame);
        end
    
end
view(-160,10)
axis on
current_frame = getframe(gcf); %gca does just the plot
writeVideo(vout,current_frame);
vout.close

deltat = floor((size(dataCase1,3))/size(A,1));

ii = 0;
trigger = 1;

if video
    video_filename = ['case1_growth_' datestr(now,'YYYYMMDD_hhmmss') '.mp4'];
    vout = VideoWriter(video_filename,'MPEG-4');
    vout.Quality = 100;
    vout.FrameRate = 10;
    vout.open;
end

for jj = 1:size(dataCase1,3)

    if jj >1
        trigger = mod(jj-1,deltat);
    end
    
    if trigger == 0
        ii = (jj-1)/deltat;
        Atemp = squeeze(A(ii,:,:));
        % Plot neural net stuff
        if ii > 1
            delete(hNNA)
        end
        hNNA = surf(gNNA.xs{1}, gNNA.xs{2}, Atemp'+.1);
        hNNA.EdgeColor = 'none';
        hNNA.FaceColor = 'b';
        hNNA.FaceAlpha = 1;
        
        
       
        %pause
    end
    
    if jj > 1
        delete(hR)
    end
    
    % Plot HJ stuff
    hR = surf(g.xs{1}, g.xs{2}, dataCase1(:,:,jj));
    hR.EdgeColor = 'none';
    hR.FaceAlpha = .4;
    hR.FaceColor = 'r';
    
    
    
    if ii
        %l=legend([hR, h0, hObs1],...
        %    {'HJ Reachability','Goal','Obstacles'});
        %l=legend([hNNA, h0, hObs1],...
        %    {'Our Algorithm','Goal','Obstacles'});
        l=legend([hR hNNA, h0, hObs1],...
            {'HJ Reachability','Our Algorithm','Goal','Obstacles'});

        l.Position = [0.5989 0.2941 0.3943 0.1738];
        l.FontSize = 20;
        %l.Location = 'northeast';
        
    end
    
    if jj == 1
        c = camlight;
        c.Position = [-3 -1 -1];
        c2 = camlight;
        c2.Position = [0 0 2];
        c3 = camlight;
        c3.Position = [2 3 0];
        view(-160,10)
    end
    
    lighting phong
    
    if video
        current_frame = getframe(gcf); %gca does just the plot
        writeVideo(vout,current_frame);
    end
end
vout.close

if video
    video_filename = ['case1_rotateback_' datestr(now,'YYYYMMDD_hhmmss') '.mp4'];
    vout = VideoWriter(video_filename,'MPEG-4');
    vout.Quality = 100;
    vout.FrameRate = 80;
    vout.open;
end


legend off
hNNAnew = visSetIm(gNNA, -Atemp', 'b', 0);
hNNAnew.LineWidth = 2;
hRnew = visSetIm(g, dataCase1(:,:,end));
hRnew.LineWidth = 2;

if video
    current_frame = getframe(gcf); %gca does just the plot
    writeVideo(vout,current_frame);
end

azRange = [-180 -160];
elRange = [90 10];

length = 160;
azStep = (azRange(1)-azRange(2))/length;
elStep = (elRange(1)-elRange(2))/length;

az = -160;
el = 10;
az = azRange(2);
el = elRange(2);
axis off

c.Visible = 'off';
c1.Visible = 'off';
c2.Visible = 'off';
delete(hR)
delete(hNNA)

while az > azRange(1) && el < elRange(1)
    az = az + azStep;
    el = el + elStep;
    view(az,el)
        if video
        current_frame = getframe(gcf); %gca does just the plot
        writeVideo(vout,current_frame);
        end
    
end
view(-180,90)
axis on
current_frame = getframe(gcf); %gca does just the plot
writeVideo(vout,current_frame);
vout.close



%% Case 2
figure(2)
clf

if video
    video_filename = ['case2_' datestr(now,'YYYYMMDD_hhmmss') '.mp4'];
    vout = VideoWriter(video_filename,'MPEG-4');
    vout.Quality = 100;
    vout.FrameRate = 10;
    vout.open;
end

% Make circle obstacles
r = .5;
x1 = (1+.5/sqrt(2));
x2 = x1;
y1 = (1+.5/sqrt(2));
y2 = -y1;
ang=0:0.01:2*pi;
xp=r*cos(ang);
yp=r*sin(ang);
hObs1 = fill(x1+xp,y1+yp,'k');
hold on
hObs2 = fill(x2+xp,y2+yp,'k');

% make square obstacle
x = [-3 -1 -1 -3];
y = [-1 -1 1 1];
hObs3 = fill(x,y,'k');

% make target set
data0x = [-1 1 1 -1];
data0y = [-1 -1 1 1];
h0 = plot(data0x,data0y);


hObs.LineWidth = 2;


h0.LineWidth = 2;
h0.LineStyle = '--';
h0.Color = 'k';
    
axis([-3 3 -3 3])
axis square
xlabel('$x$','interpreter','latex');
ylabel('$y$','interpreter','latex');
title('$u_1 \in [-1,-0.8], u_2 \in [-1, 1]$','interpreter','latex');
set(gca,'FontSize',25)
set(gcf,'Color','w')
box on

deltat = floor((size(dataCase2,3))/size(B,1));

ii = 0;
trigger = 1;
for jj = 1:size(dataCase2,3)

    if jj >1
        trigger = mod(jj-1,deltat);
    end
    
    if trigger == 0
        ii = (jj-1)/deltat;
        Btemp = squeeze(B(ii,:,:));
        % Plot neural net stuff
        if ii > 1
            delete(hNNB)
        end
        hNNB = visSetIm(gNNB, -Btemp', 'b', 0);
        hNNB.LineWidth = 2;
       
        %pause
    end
    
    if jj > 1
        delete(hR)
    end
    
    % Plot HJ stuff
    hR = visSetIm(g, dataCase2(:,:,end));
    hR.LineWidth = 2;
    
    
    if ii
        %l=legend([hR, h0, hObs1],...
        %    {'HJ Reachability','Goal','Obstacles'});
        %l=legend([hNNB, h0, hObs1],...
        %    {'Our Algorithm','Goal','Obstacles'});
        l=legend([hR hNNB, h0, hObs1],...
            {'HJ Reachability','Our Algorithm','Goal','Obstacles'});
        l.FontSize = 20;
        l.Location = 'northwest';
    end
    
    if video
        current_frame = getframe(gcf); %gca does just the plot
        writeVideo(vout,current_frame);
    end
end

%hNNA = visSetIm(gNNA, -Atemp', 'b', 0);
        %hNNA.LineWidth = 2;
        %hR = visSetIm(g, dataCase1(:,:,end));
    %hR.LineWidth = 2;
vout.close


end