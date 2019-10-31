import React from 'react'
import { Link } from 'gatsby'
import moment from 'moment'
import './style.scss'

class Post extends React.Component {
  render() {
    const { fileAbsolutePath } = this.props.data.node
    const postPath = fileAbsolutePath.match(/pages\/articles\/[\w|\W]{1,}\/(?=[2|index.md])/);
     
    let thumbnail
    const defaultThumbnailPath = '';

    try {
      thumbnail = postPath && require('../../' + postPath + 'image1.png');
    } catch{
      thumbnail = defaultThumbnailPath;
    }

    const {
      title,
      date,
      category,
      description,
    } = this.props.data.node.frontmatter
    const { slug, categorySlug } = this.props.data.node.fields
    const excerpt = this.props.data.node.excerpt;

    return (
      <div className="post">
        <div className="post__thumbnail">
          <Link className="post__title-link" to={slug}>
            <img src={thumbnail} />
          </Link>
        </div>
        <div className="post__meta">
          <time
            className="post__meta-time"
            dateTime={moment(date).format('MMMM D, YYYY')}
          >
            {moment(date).format('MMMM D, YYYY')}
          </time>
          <span className="post__meta-divider" />
          <span className="post__meta-category" key={categorySlug}>
            <Link to={categorySlug} className="post__meta-category-link">
              {category}
            </Link>
          </span>
        </div>
        <h2 className="post__title">
          <Link className="post__title-link" to={slug}>
            {title}
          </Link>
        </h2>
        <Link to={slug}>
          <span className="post__description">{excerpt}</span>
        </Link>
        {/*
        <Link className="post__readmore" to={slug}>
          Read
        </Link>
        */}
      </div>
    )
  }
}

export default Post
