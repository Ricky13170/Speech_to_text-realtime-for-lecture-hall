export function exportSRT(transcripts) {
    if (!transcripts || transcripts.length === 0) {
        alert('No transcripts to export');
        return;
    }

    let srt = '';
    let time = 0;
    const segDuration = 5;

    transcripts.forEach((t, i) => {
        const startH = Math.floor(time / 3600);
        const startM = Math.floor((time % 3600) / 60);
        const startS = Math.floor(time % 60);

        const endTime = time + segDuration;
        const endH = Math.floor(endTime / 3600);
        const endM = Math.floor((endTime % 3600) / 60);
        const endS = Math.floor(endTime % 60);

        const formatTime = (h, m, s) =>
            `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},000`;

        const start = formatTime(startH, startM, startS);
        const end = formatTime(endH, endM, endS);

        const text1 = t.vi || t.source || '';
        const text2 = t.en || t.target || '';

        srt += `${i + 1}\n`;
        srt += `${start} --> ${end}\n`;
        srt += `${text1}\n`;
        if (text2) srt += `${text2}\n`;
        srt += '\n';

        time += segDuration;
    });

    downloadFile('transcript.srt', srt, 'text/plain');
}

export function exportVTT(transcripts) {
    if (!transcripts || transcripts.length === 0) {
        alert('No transcripts to export');
        return;
    }

    let vtt = 'WEBVTT\n\n';
    let time = 0;
    const segDuration = 5;

    transcripts.forEach((t, i) => {
        const formatTime = (seconds) => {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.000`;
        };

        const start = formatTime(time);
        const end = formatTime(time + segDuration);

        const text1 = t.vi || t.source || '';
        const text2 = t.en || t.target || '';

        vtt += `${i + 1}\n`;
        vtt += `${start} --> ${end}\n`;
        vtt += `${text1}\n`;
        if (text2) vtt += `${text2}\n`;
        vtt += '\n';

        time += segDuration;
    });

    downloadFile('transcript.vtt', vtt, 'text/vtt');
}

export function exportJSON(transcripts) {
    if (!transcripts || transcripts.length === 0) {
        alert('No transcripts to export');
        return;
    }

    const data = JSON.stringify(transcripts, null, 2);
    downloadFile('transcript.json', data, 'application/json');
}

export function exportTXT(transcripts) {
    if (!transcripts || transcripts.length === 0) {
        alert('No transcripts to export');
        return;
    }

    const lines = transcripts.map(t => {
        const vi = t.vi || t.source || '';
        const en = t.en || t.target || '';
        return en ? `${vi}\n> ${en}` : vi;
    });

    downloadFile('transcript.txt', lines.join('\n\n'), 'text/plain');
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
