import React from 'react'
import Helmet from 'react-helmet'
import { graphql } from 'gatsby'
import Layout from '../components/Layout'
import Post from '../components/Post'
import Sidebar from '../components/Sidebar'

import config from '../../config';

class IndexRoute extends React.Component {
  render() {
    const items = []
    const { title, subtitle } = this.props.data.site.siteMetadata
    const posts = this.props.data.allMarkdownRemark.edges
    posts.forEach(post => {
      items.push(<Post data={post} key={post.node.fields.slug} />)
    })

    return (
      <Layout>
        <div>
          <Helmet 
            title={title}
            meta={[
              { name: 'description', content: config.description },
              { name: 'og:type', content: 'website' },
              { name: 'og:title', content: config.title },
              { name: 'og:description', content: config.description },
              { name: 'og:image', content: config.titleLogo() },
              { name: 'og:url', content: siteUrl },
            ]}>
            {/* html lang set */}
            <html lang="ko" />
            {/* load google font */}
            <link href={`https://fonts.googleapis.com/css?family=${googleFontString}`} rel="stylesheet" />
            {/* Global Site Tag (gtag.js) - Google Analytics */}
            <script async src={`https://www.googletagmanager.com/gtag/js?id=${config.googleAnalyticsTrackingId}`} />
            <script>
              {`
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', '${config.googleAnalyticsTrackingId}');
              `}
            </script>
          </Helmet>
          <Sidebar {...this.props} />
          <div className="content">
            <div className="content__inner">{items}</div>
          </div>
        </div>
      </Layout>
    )
  }
}

export default IndexRoute

export const pageQuery = graphql`
  query IndexQuery {
    site {
      siteMetadata {
        title
        subtitle
        copyright
        menu {
          label
          path
        }
        author {
          name
          email
          telegram
          twitter
          github
          rss
          vk
        }
      }
    }
    allMarkdownRemark(
      limit: 1000
      filter: { frontmatter: { layout: { eq: "post" }, draft: { ne: true } } }
      sort: { order: DESC, fields: [frontmatter___date] }
    ) {
      edges {
        node {
          fields {
            slug
            categorySlug
          }
          frontmatter {
            title
            date
            category
            description
          }
        }
      }
    }
  }
`
