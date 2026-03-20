import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import {
  ArrowLeft,
  Save,
  RotateCcw,
  Monitor,
  Target,
  Eye,
  Crosshair,
  Layers,
  Info,
  Gamepad2,
  ChevronRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DEFAULT_SETTINGS = {
  confidence_threshold: 0.5,
  aim_sensitivity: 0.8,
  target_classes: ["person"],
  enabled: true,
  show_boxes: true,
  show_crosshair: true,
  aim_assist_enabled: false,
  capture_monitor: 1,
  capture_region: null
};

const YOLO_CLASSES = [
  "person", "car", "truck", "bus", "motorcycle", "bicycle",
  "dog", "cat", "bird", "horse", "sheep", "cow", "elephant",
  "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush"
];

export default function Settings() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [availableClasses, setAvailableClasses] = useState(YOLO_CLASSES);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    loadSettings();
    loadYoloClasses();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings`);
      setSettings(response.data);
    } catch (error) {
      console.error("Failed to load settings:", error);
      toast.error("Failed to load settings");
    }
  };

  const loadYoloClasses = async () => {
    try {
      const response = await axios.get(`${API}/yolo/classes`);
      if (response.data.classes && response.data.classes.length > 0) {
        setAvailableClasses(response.data.classes);
      }
    } catch (error) {
      console.log("Using default YOLO classes");
    }
  };

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const saveSettings = async () => {
    setIsSaving(true);
    try {
      await axios.put(`${API}/settings`, settings);
      toast.success("Settings saved successfully");
      setHasChanges(false);
    } catch (error) {
      console.error("Failed to save settings:", error);
      toast.error("Failed to save settings");
    } finally {
      setIsSaving(false);
    }
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    setHasChanges(true);
    toast.info("Settings reset to defaults");
  };

  const toggleTargetClass = (className) => {
    const current = settings.target_classes || [];
    let newClasses;
    
    if (current.includes(className)) {
      newClasses = current.filter(c => c !== className);
    } else {
      newClasses = [...current, className];
    }
    
    updateSetting("target_classes", newClasses);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-4 lg:p-8" data-testid="settings-page">
      {/* Header */}
      <header className="flex items-center justify-between mb-8" data-testid="settings-header">
        <div className="flex items-center gap-4">
          <Link to="/">
            <Button variant="outline" size="icon" className="border-white/20 hover:border-[#00F0FF]/50" data-testid="back-button">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl lg:text-3xl tracking-widest text-glow-cyan">
              SETTINGS
            </h1>
            <p className="text-xs text-zinc-500 uppercase tracking-wider">
              Configure Detection Parameters
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {hasChanges && (
            <Badge variant="secondary" className="uppercase text-xs animate-pulse">
              Unsaved Changes
            </Badge>
          )}
          <Button
            variant="outline"
            onClick={resetSettings}
            className="border-white/20 hover:border-[#FF003C]/50"
            data-testid="reset-button"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          <Button
            onClick={saveSettings}
            disabled={isSaving || !hasChanges}
            className="btn-tactical"
            data-testid="save-button"
          >
            <Save className="w-4 h-4 mr-2" />
            {isSaving ? "Saving..." : "Save"}
          </Button>
        </div>
      </header>

      {/* Settings Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl">
        {/* Detection Settings */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6" data-testid="detection-section">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#00F0FF]">
            <Eye className="w-5 h-5" />
            Detection Settings
          </h2>
          
          <div className="space-y-6">
            {/* Enable Detection */}
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div>
                <Label className="text-sm">Enable Detection</Label>
                <p className="text-xs text-zinc-500 mt-1">Turn YOLO detection on/off</p>
              </div>
              <Switch
                checked={settings.enabled}
                onCheckedChange={(v) => updateSetting("enabled", v)}
                data-testid="enable-detection-toggle"
              />
            </div>
            
            {/* Confidence Threshold */}
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="text-sm">Confidence Threshold</Label>
                <span className="font-mono text-[#00F0FF]" data-testid="conf-threshold-value">
                  {(settings.confidence_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[settings.confidence_threshold * 100]}
                onValueChange={([v]) => updateSetting("confidence_threshold", v / 100)}
                max={95}
                min={10}
                step={5}
                data-testid="conf-threshold-slider"
              />
              <p className="text-xs text-zinc-500">
                Minimum confidence required for a detection to be shown
              </p>
            </div>
            
            {/* Show Bounding Boxes */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm">Show Bounding Boxes</Label>
                <p className="text-xs text-zinc-500 mt-1">Display detection rectangles</p>
              </div>
              <Switch
                checked={settings.show_boxes}
                onCheckedChange={(v) => updateSetting("show_boxes", v)}
                data-testid="show-boxes-toggle"
              />
            </div>
            
            {/* Show Crosshair */}
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm">Show Crosshair</Label>
                <p className="text-xs text-zinc-500 mt-1">Display center crosshair overlay</p>
              </div>
              <Switch
                checked={settings.show_crosshair}
                onCheckedChange={(v) => updateSetting("show_crosshair", v)}
                data-testid="show-crosshair-toggle"
              />
            </div>
          </div>
        </section>

        {/* Aim Assist Settings */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6" data-testid="aim-section">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#FF2A6D]">
            <Crosshair className="w-5 h-5" />
            Aim Assist Settings
          </h2>
          
          <div className="space-y-6">
            {/* Enable Aim Assist */}
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div>
                <Label className="text-sm">Enable Aim Assist</Label>
                <p className="text-xs text-zinc-500 mt-1">Show target lock indicator</p>
              </div>
              <Switch
                checked={settings.aim_assist_enabled}
                onCheckedChange={(v) => updateSetting("aim_assist_enabled", v)}
                data-testid="aim-assist-toggle"
              />
            </div>
            
            {/* Aim Sensitivity */}
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="text-sm">Aim Sensitivity</Label>
                <span className="font-mono text-[#FF2A6D]" data-testid="aim-sens-value">
                  {(settings.aim_sensitivity * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[settings.aim_sensitivity * 100]}
                onValueChange={([v]) => updateSetting("aim_sensitivity", v / 100)}
                max={100}
                min={10}
                step={5}
                data-testid="aim-sens-slider"
              />
              <p className="text-xs text-zinc-500">
                How quickly the aim assist responds to targets
              </p>
            </div>
            
            {/* Warning */}
            <div className="bg-[#FF003C]/10 border border-[#FF003C]/30 rounded-sm p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-[#FF003C] flex-shrink-0" />
                <p className="text-xs text-zinc-400">
                  Aim assist only calculates and displays target coordinates. 
                  Actual mouse/controller movement must be implemented separately.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Capture Settings */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6" data-testid="capture-section">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#FFD300]">
            <Monitor className="w-5 h-5" />
            Capture Settings
          </h2>
          
          <div className="space-y-6">
            {/* Monitor Selection */}
            <div className="space-y-2">
              <Label className="text-sm">Capture Monitor</Label>
              <Select
                value={String(settings.capture_monitor)}
                onValueChange={(v) => updateSetting("capture_monitor", parseInt(v))}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="monitor-select">
                  <SelectValue placeholder="Select monitor" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Monitor 1 (Primary)</SelectItem>
                  <SelectItem value="2">Monitor 2</SelectItem>
                  <SelectItem value="3">Monitor 3</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">
                Select which monitor to capture (for Xbox Remote Play window)
              </p>
            </div>
            
            {/* Region Settings Info */}
            <div className="bg-white/5 rounded-sm p-4">
              <h3 className="text-sm font-bold mb-2 flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#00F0FF]" />
                Capture Region
              </h3>
              <p className="text-xs text-zinc-400">
                Currently capturing full monitor. Custom region capture coming soon.
                For best results, run Xbox Remote Play in fullscreen mode.
              </p>
            </div>
          </div>
        </section>

        {/* Target Classes */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6" data-testid="targets-section">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#39FF14]">
            <Target className="w-5 h-5" />
            Target Classes
          </h2>
          
          <div className="space-y-4">
            <p className="text-xs text-zinc-400">
              Select which object types to detect. "person" is recommended for enemy detection.
            </p>
            
            {/* Common Classes */}
            <div>
              <Label className="text-xs text-zinc-500 uppercase mb-2 block">Common Targets</Label>
              <div className="flex flex-wrap gap-2">
                {["person", "car", "truck", "motorcycle"].map(cls => (
                  <Badge
                    key={cls}
                    variant={settings.target_classes?.includes(cls) ? "default" : "outline"}
                    className={`cursor-pointer transition-colors ${
                      settings.target_classes?.includes(cls)
                        ? "bg-[#39FF14] text-black hover:bg-[#39FF14]/80"
                        : "border-white/20 hover:border-[#39FF14]/50"
                    }`}
                    onClick={() => toggleTargetClass(cls)}
                    data-testid={`target-class-${cls}`}
                  >
                    {cls}
                  </Badge>
                ))}
              </div>
            </div>
            
            {/* All Classes */}
            <div>
              <Label className="text-xs text-zinc-500 uppercase mb-2 block">All Available Classes</Label>
              <div className="max-h-40 overflow-y-auto p-2 bg-black/50 rounded-sm">
                <div className="flex flex-wrap gap-1">
                  {availableClasses.slice(0, 30).map(cls => (
                    <Badge
                      key={cls}
                      variant={settings.target_classes?.includes(cls) ? "default" : "outline"}
                      className={`cursor-pointer text-xs transition-colors ${
                        settings.target_classes?.includes(cls)
                          ? "bg-[#39FF14]/80 text-black"
                          : "border-white/10 text-zinc-500 hover:border-white/30"
                      }`}
                      onClick={() => toggleTargetClass(cls)}
                    >
                      {cls}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Selected count */}
            <div className="flex items-center justify-between pt-2 border-t border-white/10">
              <span className="text-xs text-zinc-500">Selected targets:</span>
              <span className="font-mono text-[#39FF14]" data-testid="selected-count">
                {settings.target_classes?.length || 0}
              </span>
            </div>
          </div>
        </section>

        {/* Controller Settings Link */}
        <section className="bg-[#0A0A0A] border border-[#FF2A6D]/30 rounded-sm p-6 lg:col-span-2" data-testid="controller-link-section">
          <Link to="/controller" className="block">
            <div className="flex items-center justify-between hover:bg-white/5 p-4 -m-4 rounded-sm transition-colors">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-[#FF2A6D]/20 rounded-sm flex items-center justify-center">
                  <Gamepad2 className="w-6 h-6 text-[#FF2A6D]" />
                </div>
                <div>
                  <h2 className="text-lg uppercase tracking-wider text-[#FF2A6D]">
                    Controller Setup
                  </h2>
                  <p className="text-sm text-zinc-400">
                    Configure Scuf Valor Pro button bindings and aim assist triggers
                  </p>
                </div>
              </div>
              <ChevronRight className="w-6 h-6 text-zinc-500" />
            </div>
          </Link>
        </section>
      </div>
    </div>
  );
}
