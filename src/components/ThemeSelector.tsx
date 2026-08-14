import React from 'react';
import { useTheme } from '../context/ThemeContext';
import { ThemeMode } from '../types/theme';

interface ThemeOption {
  value: ThemeMode;
  label: string;
  description: string;
  icon: React.ReactNode;
}

const themeOptions: ThemeOption[] = [
  {
    value: 'light',
    label: 'Light',
    description: 'Always use light theme',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
  },
  {
    value: 'dark',
    label: 'Dark',
    description: 'Always use dark theme',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
      </svg>
    ),
  },
  {
    value: 'system',
    label: 'System',
    description: 'Sync with device settings',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
];

export const ThemeSelector: React.FC = () => {
  const { theme, setTheme } = useTheme();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {themeOptions.map((option) => {
        const isSelected = theme === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => setTheme(option.value)}
            className={`flex flex-col items-center justify-between p-4 rounded-xl border-2 transition-all text-left ${
              isSelected
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/30 text-blue-900 dark:text-blue-100'
                : 'border-gray-200 dark:border-neutral-800 hover:border-gray-300 dark:hover:border-neutral-700 bg-white dark:bg-neutral-900 text-gray-700 dark:text-neutral-300'
            }`}
          >
            <div className="flex items-center gap-2 mb-2 w-full">
              <span className={isSelected ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500 dark:text-neutral-400'}>
                {option.icon}
              </span>
              <span className="font-semibold text-sm">{option.label}</span>
            </div>
            <p className="text-xs text-gray-500 dark:text-neutral-400 w-full">
              {option.description}
            </p>
          </button>
        );
      })}
    </div>
  );
};
