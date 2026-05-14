export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:bg-black">
      {/* Hero Section with Video Background */}
      <div className="relative min-h-[85vh] flex items-center justify-center overflow-hidden">
        {/* Video Background */}
        <div className="absolute inset-0 w-full h-full">
          <video
            className="absolute inset-0 w-full h-full object-cover"
            autoPlay
            loop
            muted
            playsInline
          >
            <source src="/ssgrow-v2.mp4" type="video/mp4" />
            <source src="/ssgrow-v2.mp4" type="video/quicktime" />
          </video>

          {/* Dark overlay for better text readability */}
          <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/50 to-black/70 dark:from-black/70 dark:via-black/60 dark:to-black/80"></div>

          {/* Colored gradient overlay for brand effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/30 via-transparent to-teal-900/30 mix-blend-overlay"></div>
        </div>

        {/* Hero Content */}
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-28">
          <div className="text-center space-y-8">
            {/* Badge */}
            <div className="inline-block animate-fade-in">
              <span className="backdrop-blur-md bg-white/10 dark:bg-white/5 border border-white/20 text-white px-6 py-3 rounded-full text-sm font-medium shadow-lg">
                🌱 Agriculture at your fingertips
              </span>
            </div>

            {/* Main Heading */}
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-white mb-6 animate-slide-up drop-shadow-2xl">
              Empowering Farmers with Knowledge
            </h1>

            {/* Subheading */}
            <p className="text-xl lg:text-2xl text-white/90 max-w-3xl mx-auto leading-relaxed drop-shadow-lg">
              Access comprehensive agricultural laws, best practices, and
              resources to grow your farm and maximize profits
            </p>

            {/* CTA Buttons */}

            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-6">
              <a
                href="/ai-grow"
                className="bg-gradient-to-r from-emerald-500 to-teal-600 text-white px-8 py-4 rounded-lg font-semibold shadow-2xl hover:shadow-emerald-500/50 hover:scale-105 transition-all duration-300"
              >
                Open Disease AI
              </a>
              <a
                href="/suggestion-ai"
                className="bg-white/12 backdrop-blur-md border-2 border-sky-300/50 text-white px-8 py-4 rounded-lg font-semibold shadow-2xl hover:bg-sky-400/20 hover:scale-105 transition-all duration-300"
              >
                Open Suggestion AI
              </a>
              <button className="backdrop-blur-md bg-white/10 hover:bg-white/20 border-2 border-white/30 text-white px-8 py-4 rounded-lg font-semibold transition-all duration-300 hover:scale-105">
                Watch Demo
              </button>
            </div>

            {/* Stats or Trust Indicators */}
            {/* <div className="grid grid-cols-3 gap-8 max-w-2xl mx-auto pt-12">
              <div className="backdrop-blur-md bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="text-3xl font-bold text-white mb-1">500+</div>
                <div className="text-sm text-white/70">Laws & Regulations</div>
              </div>
              <div className="backdrop-blur-md bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="text-3xl font-bold text-white mb-1">50K+</div>
                <div className="text-sm text-white/70">Active Farmers</div>
              </div>
              <div className="backdrop-blur-md bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="text-3xl font-bold text-white mb-1">24/7</div>
                <div className="text-sm text-white/70">Support</div>
              </div>
            </div> */}
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-10 animate-bounce">
          <svg
            className="w-6 h-6 text-white/70"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-white dark:bg-black py-20 lg:py-28 border-t border-gray-100 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 lg:mb-20">
            <h2 className="section-title dark:text-slate-100">
              Powerful Features for Farmers
            </h2>
            <p className="section-subtitle dark:text-gray-400">
              Everything you need to succeed in modern agriculture
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-10">
            {/* Feature 1 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">📚</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Comprehensive Laws
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Access updated agricultural laws, regulations, and government
                schemes in one place
              </p>
            </div>

            {/* Feature 2 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">👤</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Profile Management
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Manage your farm profile, track your yield, and store important
                agricultural data
              </p>
            </div>

            {/* Feature 3 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Activity Tracking
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Monitor your usage history and get personalized recommendations
                based on your needs
              </p>
            </div>

            {/* Feature 4 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-emerald-600 to-green-700 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">🔒</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Secure & Private
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Your data is encrypted and protected with enterprise-grade
                security measures
              </p>
            </div>

            {/* Feature 5 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-teal-600 to-cyan-700 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">📱</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Mobile Friendly
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Access all features on your mobile device with our responsive
                design
              </p>
            </div>

            {/* Feature 6 */}
            <div className="card-hover p-8 dark:bg-slate-700/50 dark:border-slate-600">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-600 to-blue-700 rounded-xl flex items-center justify-center mb-6">
                <span className="text-2xl">🚀</span>
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
                Fast & Reliable
              </h3>
              <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                Lightning-fast performance with 99.9% uptime guarantee for your
                peace of mind
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-700 dark:to-teal-700 py-16 lg:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6">
          <h2 className="text-4xl lg:text-5xl font-bold text-white">
            Ready to Transform Your Farming?
          </h2>
          <p className="text-xl text-emerald-50 max-w-2xl mx-auto">
            Join thousands of farmers already using SS Argitech to make better
            decisions
          </p>
          <a
            href="/register"
            className="inline-block bg-white dark:bg-gray-800 text-emerald-600 dark:text-emerald-400 px-8 py-4 rounded-lg font-bold text-lg hover:bg-emerald-50 dark:hover:bg-gray-700 transition-all duration-300 hover:scale-105"
          >
            Get Started Now 🚀
          </a>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 dark:bg-slate-950 text-white py-12 border-t border-gray-800 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div>
              <h3 className="font-bold mb-4">SGrow</h3>
              <p className="text-gray-400 dark:text-gray-500 text-sm">
                Empowering farmers with knowledge and resources
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400 dark:text-gray-500 text-sm">
                <li>
                  <a href="#" className="hover:text-white transition">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    Pricing
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    FAQ
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400 dark:text-gray-500 text-sm">
                <li>
                  <a href="#" className="hover:text-white transition">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    Blog
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    Contact
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-gray-400 dark:text-gray-500 text-sm">
                <li>
                  <a href="#" className="hover:text-white transition">
                    Privacy
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    Terms
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-white transition">
                    Security
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 dark:border-slate-800 pt-8 text-center text-gray-400 dark:text-gray-500">
            <p>
              &copy; 2026 SGrow. All rights reserved. | Made with 🌾 for farmers
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
