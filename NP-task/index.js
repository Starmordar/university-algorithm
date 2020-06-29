"use strict";

function hamiltonianCycle(vertexes, start) {
    let n = vertexes.length;

    let paths = [[start]];

    while (paths.length > 0) {
        let tempPath = [];

        for (let path of paths) {
            const nextSteps = vertexes
                .find(({ vertex }) => vertex == path[path.length - 1])
                .peers
                .filter(v => !path.includes(v));

            if (!nextSteps.length) continue;

            else if (path.length == n - 1) return path.length + 1;
            else nextSteps.forEach(step => tempPath.push([...path, step]));
        }
        paths = tempPath;
    }
    return 0;
}

function hasCycle(vertexes, start, k) {
    const n = hamiltonianCycle(vertexes, start);
    return n >= k;
}

const vertexes = [
    { vertex: 1, peers: [3, 8, 15] },
    { vertex: 2, peers: [7, 14, 23] },
    { vertex: 3, peers: [1, 6, 13, 22] },
    { vertex: 4, peers: [5, 12, 21] },
    { vertex: 5, peers: [4, 11, 20] },
    { vertex: 6, peers: [3, 10, 19] },
    { vertex: 7, peers: [2, 9, 18] },
    { vertex: 8, peers: [1, 17] },
    { vertex: 9, peers: [7, 16] },
    { vertex: 10, peers: [6, 15] },
    { vertex: 11, peers: [5, 14] },
    { vertex: 12, peers: [4, 13] },
    { vertex: 13, peers: [3, 12, 23] },
    { vertex: 14, peers: [2, 11, 22] },
    { vertex: 15, peers: [1, 10, 21] },
    { vertex: 16, peers: [9, 20] },
    { vertex: 17, peers: [8, 19] },
    { vertex: 18, peers: [7] },
    { vertex: 19, peers: [6, 17] },
    { vertex: 20, peers: [5, 16] },
    { vertex: 21, peers: [4, 15] },
    { vertex: 22, peers: [3, 14] },
    { vertex: 23, peers: [2, 13] }];

let result = hasCycle(vertexes, 2, 10);
console.log('result', result);