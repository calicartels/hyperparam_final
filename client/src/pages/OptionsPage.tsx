import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Eye, Github, Save, Settings } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const OptionsPage = () => {
  const { toast } = useToast();
  
  const [settings, setSettings] = useState({
    autoActivate: true,
    showSidebar: true,
    highlightParams: true,
    showImpactLevel: true,
  });

  const handleSettingChange = (setting: string, value: boolean) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const saveSettings = () => {
    // In a real implementation, this would save to chrome.storage.sync
    chrome.storage?.sync?.set(settings, () => {
      toast({
        title: "Settings Saved",
        description: "Your preferences have been updated.",
        duration: 3000,
      });
    });
  };

  return (
    <div className="container mx-auto py-8 max-w-4xl">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold flex items-center justify-center gap-2 mb-2">
          <Eye className="h-8 w-8 text-primary" />
          <span className="bg-gradient-to-r from-primary-600 to-primary-400 bg-clip-text text-transparent">
            HyperExplainer Options
          </span>
        </h1>
        <p className="text-muted-foreground">
          Configure how HyperExplainer analyzes and displays hyperparameters
        </p>
      </header>

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-8">
          <TabsTrigger value="general">General Settings</TabsTrigger>
          <TabsTrigger value="appearance">Appearance</TabsTrigger>
          <TabsTrigger value="about">About</TabsTrigger>
        </TabsList>
        
        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>Behavior Settings</CardTitle>
              <CardDescription>
                Control how HyperExplainer works when you visit ChatGPT
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="auto-activate">Auto-activate on ChatGPT</Label>
                  <p className="text-sm text-muted-foreground">
                    Automatically activate analysis when visiting ChatGPT
                  </p>
                </div>
                <Switch 
                  id="auto-activate" 
                  checked={settings.autoActivate}
                  onCheckedChange={(checked) => handleSettingChange('autoActivate', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="show-sidebar">Show Sidebar</Label>
                  <p className="text-sm text-muted-foreground">
                    Show the explanation sidebar when analysis is active
                  </p>
                </div>
                <Switch 
                  id="show-sidebar" 
                  checked={settings.showSidebar}
                  onCheckedChange={(checked) => handleSettingChange('showSidebar', checked)}
                />
              </div>
              
              <Separator />
              
              <Button 
                className="w-full"
                onClick={saveSettings}
              >
                <Save className="h-4 w-4 mr-2" />
                Save Settings
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="appearance">
          <Card>
            <CardHeader>
              <CardTitle>Display Settings</CardTitle>
              <CardDescription>
                Customize how hyperparameters are highlighted and displayed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="highlight-params">Highlight Parameters</Label>
                  <p className="text-sm text-muted-foreground">
                    Apply background highlighting to detected hyperparameters
                  </p>
                </div>
                <Switch 
                  id="highlight-params" 
                  checked={settings.highlightParams}
                  onCheckedChange={(checked) => handleSettingChange('highlightParams', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="show-impact">Show Impact Level</Label>
                  <p className="text-sm text-muted-foreground">
                    Display impact indicators for hyperparameters
                  </p>
                </div>
                <Switch 
                  id="show-impact" 
                  checked={settings.showImpactLevel}
                  onCheckedChange={(checked) => handleSettingChange('showImpactLevel', checked)}
                />
              </div>
              
              <Separator />
              
              <Button 
                className="w-full"
                onClick={saveSettings}
              >
                <Save className="h-4 w-4 mr-2" />
                Save Settings
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="about">
          <Card>
            <CardHeader>
              <CardTitle>About HyperExplainer</CardTitle>
              <CardDescription>
                Chrome extension for explaining hyperparameters in LLM-generated code
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center space-y-4">
                <Eye className="h-16 w-16 mx-auto text-primary" />
                <h3 className="text-xl font-bold">HyperExplainer v1.0.0</h3>
                <p className="text-muted-foreground">
                  Helps users understand and customize hyperparameters in machine learning code
                  generated by ChatGPT and other LLMs.
                </p>
                
                <Separator className="my-6" />
                
                <div className="flex flex-col space-y-2">
                  <a 
                    href="https://github.com/your-repo/hyperexplainer" 
                    target="_blank" 
                    rel="noreferrer" 
                    className="flex items-center justify-center text-primary hover:underline"
                  >
                    <Github className="h-4 w-4 mr-2" />
                    GitHub Repository
                  </a>
                  <a 
                    href="https://github.com/your-repo/hyperexplainer/issues" 
                    target="_blank" 
                    rel="noreferrer" 
                    className="flex items-center justify-center text-primary hover:underline"
                  >
                    Report an Issue
                  </a>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default OptionsPage;
