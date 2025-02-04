import fs from 'fs'
import { JSDOM } from 'jsdom';
import normalizeUrl from 'normalize-url';

/**
 * Get the html source code of a page
 * 
 * @param {string} url 
 * @returns {Promise<string>} the html source code of the page
 */
async function get(url) {
    return await fetch(url).then(response => response.text());
}

/**
 * Find all the tags in the source code,
 * 
 * @param {string} source the html source code
 * @param {string} tag the html tag to find without the <>
 * @returns {object[]} an array of all the tags found, each object contains the attributes of the tag
 */
async function findTags(source, tag) {
    const dom = new JSDOM(source);
    const tags = dom.window.document.querySelectorAll(tag);

    return Array.from(tags).map(tag => {
        const attributes = {};
        for (const attr of tag.attributes) {
            attributes[attr.name] = attr.value;
        }
        return attributes;
    });
}

/**
 * 
 * @param {string} baseUrl 
 * @param {string} url 
 * @returns 
 */
function normalize(baseUrl, url) {
    if (!url) return null;

    if (baseUrl.endsWith('/')) {
        baseUrl = baseUrl.slice(0, -1);
    }

    if (url.startsWith('/')) {
        url = url.slice(1);
        baseUrl = new URL(baseUrl).origin;
    }

    url = baseUrl + '/' + url;

    return normalizeUrl(url, {
        stripWWW: false,
        stripHash: true,
        forceHttps: true,
        forceHttp: false,
        removeFragment: true,
        removeQueryParameters: true,
        removeTrailingSlash: true,
        removeSingleSlash: true,
    })
}

/**
 * Normalize the urls to be absolute, add the base url to the relative urls
 * 
 * @param {string} baseUrl 
 * @param {string[]} urls 
 */
async function normalizeUrls(baseUrl, urls) {
    // Normalize the urls using normalize-url
    return urls.map(url => normalize(baseUrl, url));
}

/**
 * Apply filters to the urls, remove the urls that are not allowed,
 * by default all the urls are rejected
 * 
 * @param {URL[]} urls 
 * @param {{ type: 'deny' | 'allow', domain: string }[]} filters 
 */
async function filterUrls(urls, filters) {
    return urls.filter(url => {
        if (!url) return false;
        if (new URL(url).pathname.includes(':')) return false;

        for (const filter of filters) {
            const domain = new URL(url).hostname;
            if (filter.type === 'allow' && domain.includes(filter.domain)) {
                return true;
            }
            if (filter.type === 'deny' && domain.includes(filter.domain)) {
                return false;
            }
        }
        return false;
    });
}

(async () => {

    const heap = ['https://www.insa-toulouse.fr'];
    const visited = new Set()

    const filters = [
        { type: 'allow', domain: 'insa-toulouse.fr' },
        { type: 'allow', domain: 'groupe-insa.fr' },
    ];

    while (heap.length > 0) {
        console.log("Heap size:", heap.length);
        console.log("Visited size:", visited.size);

        const baseUrl = heap.pop();

        console.log("Visiting:", baseUrl.toString());

        let source;

        try {
            source = await get(baseUrl)
        } catch {
            console.error('Failed to fecth: ' + baseUrl)
            continue;
        }

        const tags = await findTags(source, 'a');
        const urls = await tags.map(tag => tag.href);
    
        const normalizedUrls = await normalizeUrls(baseUrl, urls);
        
        const filteredUrls = await filterUrls(normalizedUrls, filters);

        console.log("Found urls:", filteredUrls.length);

        for (const url of filteredUrls) {
            console.log("Adding to heap:", url);
            if (!visited.has(url)) {
                heap.push(url);
                visited.add(url);
            }
        }
    }

    fs.writeFileSync('visited.txt', Array.from(visited).join('\n'));
})()