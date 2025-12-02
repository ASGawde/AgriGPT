import { Outlet, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import WeatherHeader from "@/components/WeatherHeader";

const Layout = () => {
  const location = useLocation();

  // ✅ Show weather ONLY on chat (home) page
  const showWeather = location.pathname === "/";

  return (
    <div className="min-h-screen bg-background">
      {/* Top Navigation */}
      <Navbar />

      {/* ✅ Weather Header — Chat page only */}
      {showWeather && (
        <div className="bg-gradient-to-r from-primary/5 via-transparent to-accent/5">
          <WeatherHeader />
        </div>
      )}

      {/* Main Content */}
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="relative"
      >
        {/* Subtle background pattern */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,hsl(var(--primary)/0.03)_0%,transparent_50%)] pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,hsl(var(--accent)/0.03)_0%,transparent_50%)] pointer-events-none" />

        <div className="relative">
          <Outlet />
        </div>
      </motion.main>
    </div>
  );
};

export default Layout;
