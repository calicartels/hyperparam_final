import { cn } from "@/lib/utils";
import React from "react";

interface BadgeCustomProps {
  children: React.ReactNode;
  type: 'higher' | 'lower' | 'advanced' | 'extreme';
  className?: string;
  onClick?: () => void;
}

export function BadgeCustom({ 
  children, 
  type, 
  className,
  onClick
}: BadgeCustomProps) {
  const baseClasses = "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium cursor-pointer transition-colors hover:opacity-90";
  
  let typeClasses = "";
  
  switch (type) {
    case 'higher':
      typeClasses = "bg-red-100 text-red-800 hover:bg-red-200";
      break;
    case 'lower':
      typeClasses = "bg-blue-100 text-blue-800 hover:bg-blue-200";
      break;
    case 'advanced':
      typeClasses = "bg-indigo-100 text-indigo-800 hover:bg-indigo-200";
      break;
    case 'extreme':
      typeClasses = "bg-purple-100 text-purple-800 hover:bg-purple-200";
      break;
    default:
      typeClasses = "bg-gray-100 text-gray-800 hover:bg-gray-200";
  }
  
  return (
    <span 
      className={cn(baseClasses, typeClasses, className)}
      onClick={onClick}
    >
      {children}
    </span>
  );
}