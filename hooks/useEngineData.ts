
import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchGraphData } from '../services/engineService';
import type { GraphData, EngineStatus } from '../types';

export const useEngineData = (intervalMs: number) => {
  const [data, setData] = useState<GraphData | null>(null);
  const [status, setStatus] = useState<EngineStatus>('initializing');
  const [error, setError] = useState<string | null>(null);

  const timeoutRef = useRef<number | null>(null);

  const fetchData = useCallback(async () => {
    // Clear any pending timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    try {
      const newData = await fetchGraphData();
      setData(newData);
      setStatus('online');
      setError(null);

      // Set a timeout to mark data as stale if no new data arrives
      timeoutRef.current = window.setTimeout(() => {
        setStatus('stale');
      }, intervalMs + 200);

    } catch (e) {
      console.error("Failed to fetch engine data:", e);
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred';
      setError(errorMessage);
      setStatus('error');
    }
  }, [intervalMs]);

  useEffect(() => {
    fetchData(); // Initial fetch
    const intervalId = setInterval(fetchData, intervalMs);

    return () => {
      clearInterval(intervalId);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs]);

  return { data, status, error };
};
