function visualize4D(g, data, V)
%% Obstacles

%% Neural Net Value
% V = [t, x, y, theta]

% take min Value
Vmin = V(end, :,:,:);
Vmin = squeeze(Vmin);

%% Neural Net Grid
gNN_min = [-10; -10; -pi]; % Lower corner of computation domain
gNN_max = [10; 10; pi];    % Upper corner of computation domain
N = [121; 121; 121];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
gNN = createGrid(gNN_min, gNN_max, N, pdDims);

%% Make a figure for each theta slice
thetas = [pi/4, 5*pi/4];
%thetas = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4];

for ii = 1:length(thetas)
    
    % pick figure
    figure(1)
    subplot(1,2,ii)
    
    % set theta
    theta = thetas(ii);
    
    % project NN data down to that theta
    [gNN2D, V2D] = proj(gNN, Vmin, [0 0 1], theta);
    
    % ditto with 4D HJ data. Note v = 1
    [g2D, data2D] = proj(g, data, [0 0 1 1], [theta, 1]);
    
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
    
    x = [-3 -1 -1 -3];
    y = [-1 -1 1 1];
    hObs3 = fill(x,y,'k');

    %hObs = visSetIm(g2D, -obstacles2D, 'k', 0);
    hold on
    hNN = visSetIm(gNN2D, -V2D', 'b', 0);
    hR = visSetIm(g2D, data2D, 'r', 0);
    
    data0x = [-1 1 1 -1];
    data0y = [-1 -1 1 1];
    h0 = plot(data0x,data0y);%,'g');
   
    % visualization parameters
    hR.LineWidth = 2;
    hObs.LineWidth = 2;
    hNN.LineWidth = 2;
    
    h0.LineWidth = 2;
    h0.LineStyle = '--';
    h0.Color = 'k';
    
    map = [0.0 0.0 0.0];
    hObs.Fill = 'on';
    colormap(map)
    
    %hR.Fill = 'on';
    %colormap(parula)
    
    axis([-3 3 -3 3])
    axis square
    xlabel('$x$','interpreter','latex');
    ylabel('$y$','interpreter','latex');
    title('$\theta = \pi/4$','interpreter','latex');
    set(gca,'FontSize',25)
    set(gcf,'Color','w')
    box on
end

l=legend([hR hNN, h0, hObs1],{'HJ Reachability','Our Algorithm','Goal','Obstacles'});
l.FontSize = 20;
end