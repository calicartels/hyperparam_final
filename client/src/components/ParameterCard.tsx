import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { HelpCircle } from "lucide-react";
import { BadgeCustom } from "@/components/ui/badge-custom";
import { HyperparameterInfo } from "@/lib/hyperparameters";

interface ParameterCardProps {
  paramKey: string;
  paramValue: string;
  paramInfo: HyperparameterInfo;
}

const ParameterCard: React.FC<ParameterCardProps> = ({
  paramKey,
  paramValue,
  paramInfo,
}) => {
  const renderImpactDots = (impact: string) => {
    const dotsCount = impact === "high" ? 3 : impact === "medium" ? 2 : 1;
    const dotColor = impact === "high" ? "bg-red-500" : impact === "medium" ? "bg-amber-500" : "bg-green-500";
    
    return (
      <div className="flex items-center space-x-1">
        {Array(5).fill(0).map((_, i) => (
          <div 
            key={i} 
            className={`h-2 w-2 rounded-full ${i < dotsCount ? dotColor : "bg-gray-200"}`} 
          />
        ))}
        <span className="text-xs text-gray-500 ml-2">
          {impact === "high" ? "High Impact" : impact === "medium" ? "Medium Impact" : "Low Impact"}
        </span>
      </div>
    );
  };

  const getGradientClasses = (impact: string) => {
    if (impact === "high") {
      return "from-primary-600 to-primary-500";
    } else if (impact === "medium") {
      return "from-secondary-600 to-secondary-500";
    }
    return "from-green-600 to-green-500";
  };

  return (
    <Card className="rounded-xl border border-gray-200 overflow-hidden mb-4">
      <div className={`bg-gradient-to-r ${getGradientClasses(paramInfo.impact)} px-4 py-3 text-white`}>
        <h3 className="font-medium text-lg">{paramInfo.name}</h3>
        <p className="text-sm text-white/80">Current Value: {paramValue}</p>
      </div>
      
      <CardContent className="p-4 bg-white">
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-1">Description</h4>
          <p className="text-sm text-gray-600">{paramInfo.description}</p>
        </div>
        
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-1">Impact</h4>
          {renderImpactDots(paramInfo.impact)}
        </div>

        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Alternatives</h4>
          <div className="space-y-2">
            {paramInfo.alternatives.map((alternative, index) => (
              <div 
                key={index} 
                className={`rounded-md border ${
                  alternative.type === 'advanced' || alternative.type === 'extreme' 
                    ? 'border-primary-200 bg-primary-50 hover:bg-primary-100' 
                    : 'border-gray-200 hover:border-primary-300 hover:bg-gray-50'
                } p-2 cursor-pointer transition`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-sm font-medium text-gray-700">{alternative.value}</span>
                    <p className="text-xs text-gray-500 mt-1">{alternative.description}</p>
                  </div>
                  <BadgeCustom type={alternative.type}>
                    {alternative.type.charAt(0).toUpperCase() + alternative.type.slice(1)}
                  </BadgeCustom>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
      
      <div className="bg-gray-50 px-4 py-3 border-t border-gray-200">
        <div className="flex justify-between items-center">
          <div className="text-xs text-gray-500">
            <span className="font-medium">Framework:</span> {paramInfo.framework}
          </div>
          <button className="text-xs text-primary-600 hover:text-primary-800 font-medium flex items-center">
            <HelpCircle className="h-4 w-4 mr-1" />
            Learn More
          </button>
        </div>
      </div>
    </Card>
  );
};

export default ParameterCard;
