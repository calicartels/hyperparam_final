import React from "react";
import { cn } from "@/lib/utils";

interface BadgeCustomProps {
  children: React.ReactNode;
  type: 'higher' | 'lower' | 'advanced' | 'extreme';
  className?: string;
}

export function BadgeCustom({ 
  children, 
  type, 
  className 
}: BadgeCustomProps) {
  const badgeColors = {
    higher: "bg-gray-100 text-xs text-gray-700",
    lower: "bg-gray-100 text-xs text-gray-700",
    advanced: "bg-primary-100 text-xs text-primary-700",
    extreme: "bg-secondary-100 text-xs text-secondary-700"
  };

  return (
    <div className={cn(
      "px-2 py-1 rounded",
      badgeColors[type],
      className
    )}>
      {children}
    </div>
  );
}
