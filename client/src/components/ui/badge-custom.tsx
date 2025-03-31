import * as React from "react";
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
  const badgeStyles = {
    higher: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    lower: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    advanced: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    extreme: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  };
  
  return (
    <span 
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
        badgeStyles[type],
        className
      )}
    >
      {children}
    </span>
  );
}