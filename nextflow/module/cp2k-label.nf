nextflow.enable.dsl=2

// Separate tasks give each attempt the normal site-configured CP2K resources.
// The previous task directory is an explicit, cacheable dependency of the retry.
process labelAttempt {
  tag "$name (attempt $attempt)"
  label 'cp2k'
  publishDir "$params.publish/$name/attempt-$attempt"

  input:
    tuple val(name), path(source, stageAs:'source'), path(geo, stageAs:'geometry.xyz'), val(attempt), path(helper, stageAs:'cp2k_label.py')

  output:
    tuple val(name), path('label')

  script:
    def command = params.cp2k_cmd.toString().replace("'", "'\"'\"'")
    """
    python3 '$helper' --source source --geometry geometry.xyz --attempt $attempt --command '$command'
    """
}

// Keep indices paired with their own logs when a label is omitted.
def acceptedLabels(name, idx, logs, outcomes, expected) {
  assert idx.size() == expected && idx.unique(false).size() == expected : 'Incomplete or duplicate label batch'
  def valid = (0..<idx.size()).findAll { new groovy.json.JsonSlurper().parseText(outcomes[it].text).valid }
  def required = Math.ceil(expected * 0.9).toInteger()
  assert valid.size() >= required : "Too few valid DFT labels for ${name}: ${valid.size()}/${expected}, require ${required}"
  def ordered = valid.sort {idx[it]}
  return [name, ordered.collect {idx[it]}, ordered.collect {logs[it]}]
}
