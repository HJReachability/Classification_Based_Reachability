function [g,data,tau2,time]=plane2D_reachavoid()
%% Grid
grid_min = [-4; -4]; % Lower corner of computation domain
grid_max = [4; 4];    % Upper corner of computation domain
N = [301; 301];         % Number of grid points per dimension              % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 1;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
%data0 = shapeCylinder(g, 3, [0; 4; 0], R);
data0 = shapeRectangleByCorners(g, [-1,-1], [1 1]);
% also try shapeRectangleByCorners, shapeSphere, etc.

%% time vector
t0 = 0;
tMax =2;
dt = 0.0417; %0.0333;
tau = t0:dt:tMax;

%% problem parameters

% input bounds

% Case 1
%    vxMin = -1;
%    vxMax = 1; 
%    vyMin = .5; 
%    vyMax = 1; 

% Case 2
   vxMin = -1;
   vxMax = -.8;
   vyMin = -1;
   vyMax = 1;


% control trying to min or max value function?
uMode = 'min';


%% Pack problem parameters

Plane = Plane2D([0, 0], vxMin, vxMax, vyMin,vyMax);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = Plane;
schemeData.accuracy = 'veryHigh'; %set accuracy
schemeData.uMode = uMode;


%% If you have obstacles, compute them here
obstacle1 = shapeRectangleByCorners(g, [-3, -1], [-1, 1]);
obstacle2 = -shapeRectangleByCorners(g, [-3, -3],[3, 3]);

obstacle3 = shapeSphere(g,[(1+.5/sqrt(2)),(1+.5/sqrt(2))],.5);
obstacle4 = shapeSphere(g,[(1+.5/sqrt(2)),-(1+.5/sqrt(2))],.5);
obstacles = shapeUnion(obstacle1, obstacle2);
obstacles = shapeUnion(obstacles, obstacle3);
obstacles = shapeUnion(obstacles, obstacle4);


%% Compute value function


HJIextraArgs.visualize.valueFunction = 1;
HJIextraArgs.visualize.obstaclesFunction = 1;
HJIextraArgs.visualize.initialValueFunction = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
HJIextraArgs.obstacles = obstacles;

HJIextraArgs.stopConverge = 1;   
%HJIextraArgs.keepLast = 1;

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
tic
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'zero', HJIextraArgs);
time = toc;
end