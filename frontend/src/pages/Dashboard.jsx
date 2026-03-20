import { useState, useEffect, useCallback, useRef } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import { 
  Settings, 
  Target, 
  Eye, 
  EyeOff, 
  Play, 
  Pause, 
  Crosshair,
  Activity,
  Zap,
  AlertTriangle,
  Monitor,
  RefreshCw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function Dashboard() {
  const [frame, setFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [aimTarget, setAimTarget] = useState(null);
  const [processingTime, setProcessingTime] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [settings, setSettings] = useState({
    confidence_threshold: 0.5,
    aim_sensitivity: 0.8,
    enabled: true,
    show_boxes: true,
    show_crosshair: true,
    aim_assist_enabled: false
  });
  const [fps, setFps] = useState(0);
  const [demoMode, setDemoMode] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());
  const streamIntervalRef = useRef(null);

  // Load settings
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings`);
      setSettings(response.data);
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  };

  // Calculate FPS
  const updateFps = useCallback(() => {
    frameCountRef.current++;
    const now = Date.now();
    const elapsed = now - lastFpsTimeRef.current;
    
    if (elapsed >= 1000) {
      setFps(Math.round((frameCountRef.current * 1000) / elapsed));
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }
  }, []);

  // Fetch frame
  const fetchFrame = useCallback(async () => {
    try {
      const endpoint = demoMode ? `${API}/demo/frame` : `${API}/capture`;
      const response = await axios.get(endpoint);
      
      setFrame(response.data.frame);
      setDetections(response.data.detections || []);
      setAimTarget(response.data.aim_target);
      setProcessingTime(response.data.processing_time_ms);
      setConnectionStatus("connected");
      updateFps();
    } catch (error) {
      console.error("Frame fetch error:", error);
      setConnectionStatus("error");
    }
  }, [demoMode, updateFps]);

  // Start/Stop streaming
  const toggleStream = useCallback(() => {
    if (isStreaming) {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
        streamIntervalRef.current = null;
      }
      setIsStreaming(false);
      setConnectionStatus("disconnected");
      toast.info("Stream stopped");
    } else {
      setIsStreaming(true);
      setConnectionStatus("connecting");
      fetchFrame();
      streamIntervalRef.current = setInterval(fetchFrame, 100); // ~10 FPS for demo
      toast.success("Stream started");
    }
  }, [isStreaming, fetchFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
    };
  }, []);

  // Update setting
  const updateSetting = async (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    
    try {
      await axios.put(`${API}/settings`, newSettings);
    } catch (error) {
      console.error("Failed to update settings:", error);
      toast.error("Failed to update settings");
    }
  };

  // Single capture
  const captureFrame = async () => {
    await fetchFrame();
    toast.success("Frame captured");
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected": return "bg-[#39FF14]";
      case "connecting": return "bg-[#FFD300]";
      case "error": return "bg-[#FF003C]";
      default: return "bg-zinc-500";
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-4 lg:p-6" data-testid="dashboard">
      {/* Header */}
      <header className="flex items-center justify-between mb-6" data-testid="dashboard-header">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8 text-[#00F0FF]" />
            <div>
              <h1 className="text-2xl lg:text-3xl tracking-widest text-glow-cyan">
                XBOX VISION AI
              </h1>
              <p className="text-xs text-zinc-500 uppercase tracking-wider">
                Computer Vision Aimbot System
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 ml-6">
            <div className={`status-indicator ${connectionStatus === "connected" ? "active" : "inactive"}`} />
            <span className="text-xs uppercase tracking-wider text-zinc-400">
              {connectionStatus}
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Badge 
            variant={demoMode ? "secondary" : "default"}
            className="uppercase font-mono text-xs"
            data-testid="demo-badge"
          >
            {demoMode ? "Demo Mode" : "Live Mode"}
          </Badge>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Link to="/settings">
                  <Button variant="outline" size="icon" className="border-white/20 hover:border-[#00F0FF]/50" data-testid="settings-button">
                    <Settings className="w-5 h-5" />
                  </Button>
                </Link>
              </TooltipTrigger>
              <TooltipContent>
                <p>Settings</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6">
        {/* Video Feed */}
        <div className="lg:col-span-9">
          <div className="video-feed rounded-sm relative corner-brackets" data-testid="video-feed">
            {frame ? (
              <img 
                src={`data:image/jpeg;base64,${frame}`}
                alt="Detection Feed"
                className="w-full h-full object-contain"
                data-testid="video-frame"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0A0A0A]">
                <Monitor className="w-16 h-16 text-zinc-700 mb-4" />
                <p className="text-zinc-500 uppercase tracking-wider text-sm">
                  No Feed
                </p>
                <p className="text-zinc-600 text-xs mt-2">
                  Press PLAY to start capturing
                </p>
              </div>
            )}
            
            {/* HUD Overlay */}
            <div className="absolute inset-0 pointer-events-none">
              {/* Top bar */}
              <div className="absolute top-0 left-0 right-0 p-3 flex justify-between items-start bg-gradient-to-b from-black/50 to-transparent">
                <div className="flex items-center gap-4">
                  <div className="data-card py-2 px-3">
                    <span className="text-xs text-zinc-400 uppercase">Detections</span>
                    <p className="font-mono text-xl text-[#FF003C] text-glow-red" data-testid="detection-count">
                      {detections.length}
                    </p>
                  </div>
                  
                  <div className="data-card py-2 px-3">
                    <span className="text-xs text-zinc-400 uppercase">Proc Time</span>
                    <p className="font-mono text-xl text-[#00F0FF]" data-testid="processing-time">
                      {processingTime.toFixed(1)}ms
                    </p>
                  </div>
                </div>
                
                <div className="data-card py-2 px-3 border-[#39FF14]">
                  <span className="text-xs text-zinc-400 uppercase">FPS</span>
                  <p className="font-mono text-xl text-[#39FF14]" data-testid="fps-counter">
                    {fps}
                  </p>
                </div>
              </div>
              
              {/* Detection list overlay */}
              {detections.length > 0 && (
                <div className="absolute bottom-3 left-3 max-w-xs">
                  {detections.slice(0, 3).map((det, idx) => (
                    <div 
                      key={idx}
                      className="data-card py-1 px-2 mb-1 bg-black/80 border-[#FF003C] detection-lock"
                    >
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-xs uppercase text-[#FF003C]">
                          {det.class_name}
                        </span>
                        <span className="font-mono text-xs text-zinc-400">
                          {(det.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Aim target indicator */}
              {aimTarget && settings.aim_assist_enabled && (
                <div className="absolute bottom-3 right-3">
                  <div className="data-card py-2 px-3 bg-black/80 border-[#39FF14] glow-green">
                    <div className="flex items-center gap-2">
                      <Crosshair className="w-4 h-4 text-[#39FF14]" />
                      <span className="text-xs uppercase text-[#39FF14]">Target Locked</span>
                    </div>
                    <p className="font-mono text-xs text-zinc-400 mt-1">
                      X: {aimTarget[0]} Y: {aimTarget[1]}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Control Bar */}
          <div className="mt-4 flex items-center justify-between bg-[#0A0A0A] p-4 rounded-sm border border-white/10" data-testid="control-bar">
            <div className="flex items-center gap-4">
              <Button
                onClick={toggleStream}
                className={isStreaming ? "bg-[#FF003C] hover:bg-[#FF003C]/80" : "btn-tactical"}
                data-testid="stream-toggle"
              >
                {isStreaming ? (
                  <>
                    <Pause className="w-4 h-4 mr-2" />
                    STOP
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    START
                  </>
                )}
              </Button>
              
              <Button
                variant="outline"
                onClick={captureFrame}
                className="border-white/20 hover:border-[#00F0FF]/50"
                data-testid="capture-button"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                CAPTURE
              </Button>
            </div>
            
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <span className="text-xs uppercase text-zinc-400">Demo Mode</span>
                <Switch
                  checked={demoMode}
                  onCheckedChange={setDemoMode}
                  data-testid="demo-mode-toggle"
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Sidebar */}
        <div className="lg:col-span-3 flex flex-col gap-4" data-testid="sidebar">
          {/* Quick Settings */}
          <div className="bg-[#0A0A0A] border border-white/10 rounded-sm p-4">
            <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-[#00F0FF]" />
              Quick Settings
            </h2>
            
            <div className="space-y-4">
              {/* Detection Toggle */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {settings.enabled ? (
                    <Eye className="w-4 h-4 text-[#39FF14]" />
                  ) : (
                    <EyeOff className="w-4 h-4 text-zinc-500" />
                  )}
                  <span className="text-sm">Detection</span>
                </div>
                <Switch
                  checked={settings.enabled}
                  onCheckedChange={(v) => updateSetting("enabled", v)}
                  data-testid="detection-toggle"
                />
              </div>
              
              {/* Show Boxes */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Show Boxes</span>
                <Switch
                  checked={settings.show_boxes}
                  onCheckedChange={(v) => updateSetting("show_boxes", v)}
                  data-testid="boxes-toggle"
                />
              </div>
              
              {/* Show Crosshair */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-400">Crosshair</span>
                <Switch
                  checked={settings.show_crosshair}
                  onCheckedChange={(v) => updateSetting("show_crosshair", v)}
                  data-testid="crosshair-toggle"
                />
              </div>
              
              {/* Aim Assist */}
              <div className="flex items-center justify-between pt-2 border-t border-white/10">
                <div className="flex items-center gap-2">
                  <Crosshair className="w-4 h-4 text-[#FF2A6D]" />
                  <span className="text-sm">Aim Assist</span>
                </div>
                <Switch
                  checked={settings.aim_assist_enabled}
                  onCheckedChange={(v) => updateSetting("aim_assist_enabled", v)}
                  data-testid="aim-assist-toggle"
                />
              </div>
            </div>
          </div>
          
          {/* Confidence Slider */}
          <div className="bg-[#0A0A0A] border border-white/10 rounded-sm p-4">
            <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#00F0FF]" />
              Confidence
            </h2>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Threshold</span>
                <span className="font-mono text-[#00F0FF]" data-testid="confidence-value">
                  {(settings.confidence_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[settings.confidence_threshold * 100]}
                onValueChange={([v]) => updateSetting("confidence_threshold", v / 100)}
                max={100}
                min={10}
                step={5}
                className="w-full"
                data-testid="confidence-slider"
              />
              <p className="text-xs text-zinc-600">
                Higher = fewer false positives
              </p>
            </div>
          </div>
          
          {/* Sensitivity Slider */}
          <div className="bg-[#0A0A0A] border border-white/10 rounded-sm p-4">
            <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-4 flex items-center gap-2">
              <Target className="w-4 h-4 text-[#FF2A6D]" />
              Aim Sensitivity
            </h2>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-zinc-500">Speed</span>
                <span className="font-mono text-[#FF2A6D]" data-testid="sensitivity-value">
                  {(settings.aim_sensitivity * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[settings.aim_sensitivity * 100]}
                onValueChange={([v]) => updateSetting("aim_sensitivity", v / 100)}
                max={100}
                min={10}
                step={5}
                className="w-full"
                data-testid="sensitivity-slider"
              />
            </div>
          </div>
          
          {/* Warning Card */}
          <div className="bg-[#FF003C]/10 border border-[#FF003C]/30 rounded-sm p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-[#FF003C] flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-bold text-[#FF003C] uppercase">Warning</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Using aim assist in online multiplayer games may result in account bans. Use responsibly in single-player or private matches only.
                </p>
              </div>
            </div>
          </div>
          
          {/* Detection Stats */}
          <div className="bg-[#0A0A0A] border border-white/10 rounded-sm p-4 mt-auto">
            <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-3">
              Session Stats
            </h2>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-2 bg-white/5 rounded-sm">
                <p className="font-mono text-lg text-[#00F0FF]" data-testid="total-detections">
                  {detections.length}
                </p>
                <p className="text-xs text-zinc-500 uppercase">Current</p>
              </div>
              <div className="text-center p-2 bg-white/5 rounded-sm">
                <p className="font-mono text-lg text-[#39FF14]">
                  {fps}
                </p>
                <p className="text-xs text-zinc-500 uppercase">FPS</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
