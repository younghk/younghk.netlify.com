let config = {
    title: `Cheong`,
    author: 'Cheong',
    description: "Cheong's log",
    siteUrl: 'https://cheong.netlify.com',
  
    // # Header config
    titleLogo: () => {
      return require('./src/pages/profile.png');
    },
    titleLogoShow: true,
    bio: 'Live, Lovely',
    bioShow: true,
  
    // # Addtional
    googleAnalyticsTrackingId: 'UA-136078306-2',
    disqusShortname: 'cheong-netlify-com',
  
    // ## google AdSense
    // In addition, client-id in '/static/ads.txt' file needs to be modified
    googleAdsense: false,
    adsenseClient: '',
    adsenseSlot: '',
  };

  module.exports = config;
