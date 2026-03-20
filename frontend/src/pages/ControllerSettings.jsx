import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import {
  ArrowLeft,
  Save,
  Gamepad2,
  Vibrate,
  Settings2,
  Info,
  Zap,
  Target,
  Shield,
  Hand
} from "lucide-react";
import { Button } from "@/components/ui/button";
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

const BUTTON_OPTIONS = [
  { value: "A", label: "A Button" },
  { value: "B", label: "B Button" },
  { value: "X", label: "X Button" },
  { value: "Y", label: "Y Button" },
  { value: "LEFT_SHOULDER", label: "LB (Left Bumper)" },
  { value: "RIGHT_SHOULDER", label: "RB (Right Bumper)" },
  { value: "LEFT_TRIGGER", label: "LT (Left Trigger)" },
  { value: "RIGHT_TRIGGER", label: "RT (Right Trigger)" },
  { value: "LEFT_THUMB", label: "LS (Left Stick Click)" },
  { value: "RIGHT_THUMB", label: "RS (Right Stick Click)" },
  { value: "START", label: "Start / Menu" },
  { value: "BACK", label: "Back / View" },
  { value: "DPAD_UP", label: "D-Pad Up" },
  { value: "DPAD_DOWN", label: "D-Pad Down" },
  { value: "DPAD_LEFT", label: "D-Pad Left" },
  { value: "DPAD_RIGHT", label: "D-Pad Right" },
];

const DEFAULT_BINDINGS = {
  aim_assist_toggle: "LEFT_SHOULDER",
  aim_hold: "LEFT_TRIGGER",
  fire: "RIGHT_TRIGGER",
  snap_to_target: "RIGHT_THUMB",
  cycle_target: "RIGHT_SHOULDER",
  toggle_overlay: "BACK",
};

export default function ControllerSettings() {
  const [controllerStatus, setControllerStatus] = useState({
    available: false,
    connected: false
  });
  const [bindings, setBindings] = useState(DEFAULT_BINDINGS);
  const [config, setConfig] = useState({
    enabled: true,
    trigger_threshold: 0.3,
    vibration_enabled: true,
    scuf_mode: true
  });
  const [scufConfig, setScufConfig] = useState(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    loadControllerStatus();
    loadBindings();
    loadScufConfig();
    
    // Poll controller status
    const interval = setInterval(loadControllerStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadControllerStatus = async () => {
    try {
      const response = await axios.get(`${API}/controller/status`);
      setControllerStatus({
        available: response.data.available,
        connected: response.data.connected
      });
      if (response.data.config) {
        setConfig(response.data.config);
      }
    } catch (error) {
      console.error("Failed to load controller status:", error);
    }
  };

  const loadBindings = async () => {
    try {
      const response = await axios.get(`${API}/controller/bindings`);
      setBindings(response.data.bindings || DEFAULT_BINDINGS);
    } catch (error) {
      console.error("Failed to load bindings:", error);
    }
  };

  const loadScufConfig = async () => {
    try {
      const response = await axios.get(`${API}/controller/scuf-config`);
      setScufConfig(response.data);
    } catch (error) {
      console.error("Failed to load Scuf config:", error);
    }
  };

  const updateBinding = (action, button) => {
    setBindings(prev => ({ ...prev, [action]: button }));
    setHasChanges(true);
  };

  const saveBindings = async () => {
    setIsSaving(true);
    try {
      await axios.put(`${API}/controller/bindings`, bindings);
      toast.success("Controller bindings saved");
      setHasChanges(false);
    } catch (error) {
      toast.error("Failed to save bindings");
    } finally {
      setIsSaving(false);
    }
  };

  const testVibration = async () => {
    try {
      await axios.post(`${API}/controller/vibrate?left=0.5&right=0.8&duration=300`);
      toast.success("Vibration sent!");
    } catch (error) {
      toast.error("Vibration failed - Is controller connected?");
    }
  };

  const applyScufDefaults = () => {
    if (scufConfig?.recommended_bindings) {
      setBindings(prev => ({
        ...prev,
        ...scufConfig.recommended_bindings
      }));
      setHasChanges(true);
      toast.info("Scuf Valor Pro defaults applied");
    }
  };

  const getButtonLabel = (value) => {
    return BUTTON_OPTIONS.find(b => b.value === value)?.label || value;
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-4 lg:p-8" data-testid="controller-settings">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Link to="/settings">
            <Button variant="outline" size="icon" className="border-white/20 hover:border-[#00F0FF]/50" data-testid="back-button">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl lg:text-3xl tracking-widest text-glow-cyan">
              CONTROLLER SETUP
            </h1>
            <p className="text-xs text-zinc-500 uppercase tracking-wider">
              Scuf Valor Pro Configuration
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
            onClick={saveBindings}
            disabled={isSaving || !hasChanges}
            className="btn-tactical"
            data-testid="save-button"
          >
            <Save className="w-4 h-4 mr-2" />
            {isSaving ? "Saving..." : "Save"}
          </Button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl">
        {/* Controller Status */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#00F0FF]">
            <Gamepad2 className="w-5 h-5" />
            Controller Status
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-white/5 rounded-sm">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${controllerStatus.connected ? 'bg-[#39FF14] animate-pulse' : 'bg-zinc-600'}`} />
                <div>
                  <p className="font-bold">Scuf Valor Pro</p>
                  <p className="text-xs text-zinc-400">
                    {controllerStatus.connected ? "Connected" : "Not Connected"}
                  </p>
                </div>
              </div>
              {controllerStatus.connected && (
                <Badge className="bg-[#39FF14] text-black">READY</Badge>
              )}
            </div>
            
            {!controllerStatus.available && (
              <div className="bg-[#FFD300]/10 border border-[#FFD300]/30 rounded-sm p-4">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-[#FFD300] flex-shrink-0" />
                  <div>
                    <p className="text-sm font-bold text-[#FFD300]">Windows Only</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      Controller input requires XInput (Windows). 
                      The app will work on your local gaming PC.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <Button
              onClick={testVibration}
              variant="outline"
              className="w-full border-[#FF2A6D]/50 hover:border-[#FF2A6D]"
              disabled={!controllerStatus.connected}
              data-testid="test-vibration"
            >
              <Vibrate className="w-4 h-4 mr-2" />
              Test Vibration
            </Button>
          </div>
        </section>

        {/* Scuf Valor Pro Info */}
        <section className="bg-[#0A0A0A] border border-[#FF2A6D]/30 rounded-sm p-6">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#FF2A6D]">
            <Settings2 className="w-5 h-5" />
            Scuf Valor Pro
          </h2>
          
          {scufConfig && (
            <div className="space-y-4">
              <p className="text-sm text-zinc-400">{scufConfig.description}</p>
              
              <div className="space-y-2">
                <p className="text-xs uppercase text-zinc-500">Recommended Setup:</p>
                <ul className="space-y-1">
                  {scufConfig.notes?.map((note, idx) => (
                    <li key={idx} className="text-xs text-zinc-400 flex items-start gap-2">
                      <Zap className="w-3 h-3 text-[#FF2A6D] flex-shrink-0 mt-0.5" />
                      {note}
                    </li>
                  ))}
                </ul>
              </div>
              
              <Button
                onClick={applyScufDefaults}
                className="w-full bg-[#FF2A6D] hover:bg-[#FF2A6D]/80"
                data-testid="apply-scuf-defaults"
              >
                Apply Scuf Defaults
              </Button>
            </div>
          )}
        </section>

        {/* Button Bindings */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6 lg:col-span-2">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-[#FFD300]">
            <Hand className="w-5 h-5" />
            Button Bindings
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Aim Assist Toggle */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-[#00F0FF]" />
                <Label className="text-sm">Aim Assist Toggle</Label>
              </div>
              <Select
                value={bindings.aim_assist_toggle}
                onValueChange={(v) => updateBinding("aim_assist_toggle", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-aim-assist">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Quick toggle aim assist on/off</p>
            </div>

            {/* Aim Hold (ADS) */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-[#FF2A6D]" />
                <Label className="text-sm">Aim Down Sights</Label>
              </div>
              <Select
                value={bindings.aim_hold}
                onValueChange={(v) => updateBinding("aim_hold", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-aim-hold">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Hold for aim assist while ADS</p>
            </div>

            {/* Fire */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-[#FF003C]" />
                <Label className="text-sm">Fire / Shoot</Label>
              </div>
              <Select
                value={bindings.fire}
                onValueChange={(v) => updateBinding("fire", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-fire">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Primary fire button</p>
            </div>

            {/* Snap to Target */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-[#39FF14]" />
                <Label className="text-sm">Snap to Target</Label>
              </div>
              <Select
                value={bindings.snap_to_target}
                onValueChange={(v) => updateBinding("snap_to_target", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-snap">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Instant snap aim to nearest target</p>
            </div>

            {/* Cycle Target */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-[#00F0FF]" />
                <Label className="text-sm">Cycle Target</Label>
              </div>
              <Select
                value={bindings.cycle_target}
                onValueChange={(v) => updateBinding("cycle_target", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-cycle">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Switch between detected targets</p>
            </div>

            {/* Toggle Overlay */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-zinc-400" />
                <Label className="text-sm">Toggle Overlay</Label>
              </div>
              <Select
                value={bindings.toggle_overlay}
                onValueChange={(v) => updateBinding("toggle_overlay", v)}
              >
                <SelectTrigger className="bg-black border-white/20" data-testid="binding-overlay">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BUTTON_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-zinc-500">Show/hide detection boxes</p>
            </div>
          </div>
        </section>

        {/* Advanced Settings */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6 lg:col-span-2">
          <h2 className="text-lg uppercase tracking-wider mb-6 flex items-center gap-3 text-zinc-400">
            <Settings2 className="w-5 h-5" />
            Advanced Settings
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Trigger Threshold */}
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label className="text-sm">Trigger Threshold</Label>
                <span className="font-mono text-[#00F0FF]">
                  {(config.trigger_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[config.trigger_threshold * 100]}
                onValueChange={([v]) => {
                  setConfig(prev => ({ ...prev, trigger_threshold: v / 100 }));
                  setHasChanges(true);
                }}
                max={80}
                min={10}
                step={5}
                data-testid="trigger-threshold"
              />
              <p className="text-xs text-zinc-500">
                How far trigger must be pressed to activate (Scuf trigger stops: use 20-30%)
              </p>
            </div>

            {/* Vibration */}
            <div className="flex items-center justify-between p-4 bg-white/5 rounded-sm">
              <div>
                <Label className="text-sm">Vibration Feedback</Label>
                <p className="text-xs text-zinc-500 mt-1">Haptic feedback on target lock</p>
              </div>
              <Switch
                checked={config.vibration_enabled}
                onCheckedChange={(v) => {
                  setConfig(prev => ({ ...prev, vibration_enabled: v }));
                  setHasChanges(true);
                }}
                data-testid="vibration-toggle"
              />
            </div>
          </div>
        </section>

        {/* Control Reference */}
        <section className="bg-[#0A0A0A] border border-white/10 rounded-sm p-6 lg:col-span-2">
          <h2 className="text-lg uppercase tracking-wider mb-4 text-zinc-400">
            Quick Reference
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 text-center">
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">Toggle Aim</p>
              <p className="font-bold text-[#00F0FF]">{getButtonLabel(bindings.aim_assist_toggle)}</p>
            </div>
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">ADS</p>
              <p className="font-bold text-[#FF2A6D]">{getButtonLabel(bindings.aim_hold)}</p>
            </div>
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">Fire</p>
              <p className="font-bold text-[#FF003C]">{getButtonLabel(bindings.fire)}</p>
            </div>
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">Snap</p>
              <p className="font-bold text-[#39FF14]">{getButtonLabel(bindings.snap_to_target)}</p>
            </div>
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">Cycle</p>
              <p className="font-bold text-[#00F0FF]">{getButtonLabel(bindings.cycle_target)}</p>
            </div>
            <div className="p-3 bg-white/5 rounded-sm">
              <p className="text-xs text-zinc-500">Overlay</p>
              <p className="font-bold text-zinc-400">{getButtonLabel(bindings.toggle_overlay)}</p>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-[#FF003C]/10 border border-[#FF003C]/30 rounded-sm">
            <p className="text-xs text-center text-zinc-400">
              <strong className="text-[#FF003C]">Emergency Disable:</strong> Hold LB + RB together to instantly disable all aim assist
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
